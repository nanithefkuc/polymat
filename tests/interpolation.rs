use fgf::field::{Elem, Field};
use fgf::{Gf8B, Mersenne31, gf8b, mersenne31};
use poly_ring::Polynomial;
use polymat::{InterpolationConstraint, PolynomialMatrix};

type B = <Gf8B as Field>::Elem;
type P = <Mersenne31 as Field>::Elem;

fn b(value: u8) -> B {
    gf8b::Elem::from_raw(value)
}

fn p(value: u32) -> P {
    mersenne31::Elem::from_raw(value)
}

fn poly_b(coefficients: &[B]) -> Polynomial<Gf8B> {
    Polynomial::from_coefficients(coefficients).unwrap()
}

fn poly_p(coefficients: &[P]) -> Polynomial<Mersenne31> {
    Polynomial::from_coefficients(coefficients).unwrap()
}

fn hasse_satisfies<F: fgf::kernel::FieldKernels>(
    basis: &PolynomialMatrix<F>,
    input: &PolynomialMatrix<F>,
    constraints: &[InterpolationConstraint<F>],
) -> bool {
    let (krows, _) = basis.shape();
    let (mrows, _) = input.shape();
    for k in 0..krows {
        for constraint in constraints {
            let mut inner = Polynomial::zero();
            for i in 0..mrows {
                let term = basis
                    .entry(k, i)
                    .unwrap()
                    .multiply(input.entry(i, constraint.column).unwrap())
                    .unwrap();
                inner.add_assign(&term).unwrap();
            }
            for hasse in 0..constraint.multiplicity {
                if !inner.evaluate_hasse(constraint.point, hasse).is_zero() {
                    return false;
                }
            }
        }
    }
    true
}

#[test]
fn interpolation_matches_congruence_moduli() {
    // F = [1; x], constraint (0, col 0, mult 2): same module as M = [x^2].
    let input =
        PolynomialMatrix::from_entries(2, 1, alloc::vec![poly_b(&[b(1)]), poly_b(&[b(0), b(1)])])
            .unwrap();
    let constraints = alloc::vec![InterpolationConstraint::new(b(0), 0, 2)];
    let recurrence = input.interpolation_basis(&constraints, &[0, 0]).unwrap();
    let moduli = input.interpolation_moduli(&constraints).unwrap();
    assert_eq!(moduli.len(), 1);
    let oracle = input.congruence_basis(&moduli, &[0, 0]).unwrap();
    assert_eq!(recurrence.basis, oracle);
    assert!(hasse_satisfies(&recurrence.basis, &input, &constraints));
    // All-zero points reduce to the approximant constructor.
    let approximant = input.approximant_basis(&[2], &[0, 0]).unwrap();
    assert_eq!(recurrence.basis, approximant.basis);
    assert_eq!(
        recurrence.independent_constraints,
        approximant.independent_constraints
    );
}

#[test]
fn duplicate_points_merge_by_maximum() {
    let input = PolynomialMatrix::from_entries(1, 1, alloc::vec![poly_b(&[b(1)])]).unwrap();
    let constraints = alloc::vec![
        InterpolationConstraint::new(b(1), 0, 1),
        InterpolationConstraint::new(b(1), 0, 3),
        InterpolationConstraint::new(b(1), 0, 2),
    ];
    let moduli = input.interpolation_moduli(&constraints).unwrap();
    // (x-1)^3 over GF(8): x^3 + 3x^2 + 3x + 1 with char-2 collapse = x^3+x^2+x+1.
    let linear = poly_b(&[b(1), b(1)]);
    let expected = linear.multiply(&linear).unwrap().multiply(&linear).unwrap();
    assert_eq!(moduli[0], expected);
    let result = input.interpolation_basis(&constraints, &[0]).unwrap();
    assert_eq!(result.independent_constraints, 3);
    assert!(
        input
            .interpolation_basis(&[InterpolationConstraint::new(b(0), 5, 1)], &[0])
            .is_err()
    );
    // Distinct columns at the same point stay independent.
    let two = PolynomialMatrix::from_entries(
        2,
        2,
        alloc::vec![
            poly_b(&[b(1)]),
            Polynomial::zero(),
            Polynomial::zero(),
            poly_b(&[b(1)]),
        ],
    )
    .unwrap();
    let both = alloc::vec![
        InterpolationConstraint::new(b(1), 0, 2),
        InterpolationConstraint::new(b(1), 1, 2),
    ];
    let result = two.interpolation_basis(&both, &[0, 0]).unwrap();
    assert_eq!(result.independent_constraints, 4);
    assert!(hasse_satisfies(&result.basis, &two, &both));
}

#[test]
fn multiplicity_above_characteristic_uses_hasse() {
    // GF(8) has characteristic 2; multiplicity 3 exceeds it. The Hasse
    // coefficient H_2(x^2 at 0) = 1 is nonzero, so the constraint is
    // nontrivial; an ordinary second derivative divided by 2! would read
    // 2/2 = 1 in char 0 but is undefined in char 2 (2 = 0).
    let input = PolynomialMatrix::from_entries(1, 1, alloc::vec![poly_b(&[b(1)])]).unwrap();
    let constraints = alloc::vec![InterpolationConstraint::new(b(0), 0, 3)];
    let result = input.interpolation_basis(&constraints, &[0]).unwrap();
    assert_eq!(
        result.basis.entry(0, 0).unwrap(),
        &poly_b(&[b(0), b(0), b(0), b(1)])
    );
    // Ordinary-derivative substitution fails: d^2/dx^2 (x^2) = 2 = 0 in
    // characteristic 2, so it cannot distinguish x^2 from 0 at order 2,
    // while H_2(x^2 at 0) = 1 does.
    let probe = poly_b(&[b(0), b(0), b(1)]);
    assert_eq!(probe.evaluate_hasse(b(0), 2), b(1));
    let formal_second = probe
        .hasse_derivative(1)
        .unwrap()
        .hasse_derivative(1)
        .unwrap();
    assert!(formal_second.evaluate(b(0)).is_zero());
}

#[test]
fn odd_characteristic_interpolation_agrees() {
    let input =
        PolynomialMatrix::from_entries(2, 1, alloc::vec![poly_p(&[p(1)]), poly_p(&[p(0), p(1)])])
            .unwrap();
    let constraints = alloc::vec![
        InterpolationConstraint::new(p(2), 0, 2),
        InterpolationConstraint::new(p(5), 0, 1),
    ];
    let recurrence = input.interpolation_basis(&constraints, &[0, 0]).unwrap();
    let moduli = input.interpolation_moduli(&constraints).unwrap();
    let oracle = input.congruence_basis(&moduli, &[0, 0]).unwrap();
    assert_eq!(recurrence.basis, oracle);
    assert!(hasse_satisfies(&recurrence.basis, &input, &constraints));
}

#[test]
fn scalar_pade_satisfies_equation_and_bounds() {
    // Series 1/(1-x) = 1+x+x^2+... truncated: Padé [1/1] recovers [1, 1-x].
    let series = poly_b(&[b(1), b(1), b(1), b(1)]);
    let relation = PolynomialMatrix::<Gf8B>::pade_relation(&series, 4, 1).unwrap();
    // numerator - series*denominator = 0 mod x^4.
    let product = series.multiply(&relation.denominator).unwrap();
    let mut residual = relation.numerator.clone();
    residual.sub_assign(&product).unwrap();
    for coefficient in 0..4 {
        assert!(residual.coefficient(coefficient).is_zero());
    }
    // Denominator-degenerate: order 1 with denominator degree 1 selects
    // the [0/x] row (numerator zero, denominator x), which still satisfies
    // the equation without claiming a decoder verdict.
    let degenerate = PolynomialMatrix::<Gf8B>::pade_relation(&series, 1, 1).unwrap();
    assert!(degenerate.numerator.is_zero());
    assert_eq!(degenerate.denominator, poly_b(&[b(0), b(1)]));
}

#[test]
fn simultaneous_and_hermite_pade_hold() {
    // Both series share the denominator 1+x at order 3: s0 = 1/(1+x)
    // and s1 = (1+2x)/(1+x) truncated, with numerators 1 and 1+2x. The
    // stacked module row [1, 1+2x, 1+x] is the shared relation.
    let first = poly_b(&[b(1), b(1), b(1)]);
    let second = poly_b(&[b(1), b(3), b(3)]);
    // Non-geometric normal data also selects: 1+2x+3x^2 at order 3 with
    // denominator degree 1 (its Hankel determinant is nonzero).
    let normal = poly_b(&[b(1), b(2), b(3)]);
    let relation = PolynomialMatrix::<Gf8B>::pade_relation(&normal, 3, 1).unwrap();
    let product = normal.multiply(&relation.denominator).unwrap();
    let mut residual = relation.numerator.clone();
    residual.sub_assign(&product).unwrap();
    for coefficient in 0..3 {
        assert!(residual.coefficient(coefficient).is_zero());
    }
    assert_eq!(relation.denominator.degree(), Some(1));
    let relations =
        PolynomialMatrix::<Gf8B>::simultaneous_pade(&[first.clone(), second.clone()], 3, 1)
            .unwrap();
    assert_eq!(relations.len(), 2);
    for (series, relation) in [first, second].into_iter().zip(&relations) {
        let product = series.multiply(&relation.denominator).unwrap();
        let mut residual = relation.numerator.clone();
        residual.sub_assign(&product).unwrap();
        for coefficient in 0..3 {
            assert!(residual.coefficient(coefficient).is_zero());
        }
    }
    // The denominator is genuinely shared across both relations.
    assert_eq!(relations[0].denominator, relations[1].denominator);
    // Hermite-Padé: [1, x] with bounds [1, 2] and order 2 returns a
    // relation annihilating the series: series[0]*r[0] + series[1]*r[1]
    // vanishes mod x^2.
    let series = alloc::vec![poly_b(&[b(1)]), poly_b(&[b(0), b(1)])];
    let relation = PolynomialMatrix::<Gf8B>::hermite_pade(&series, 2, &[1, 2]).unwrap();
    assert_eq!(relation.len(), 2);
    for (entry, &bound) in relation.iter().zip(&[1, 2]) {
        assert!(entry.degree().is_none_or(|degree| degree < bound));
    }
    let mut combined = Polynomial::zero();
    for (target, factor) in series.iter().zip(&relation) {
        let term = target.multiply(factor).unwrap();
        combined.add_assign(&term).unwrap();
    }
    for coefficient in 0..2 {
        assert!(combined.coefficient(coefficient).is_zero());
    }
    assert!(PolynomialMatrix::<Gf8B>::hermite_pade(&[], 2, &[]).is_err());
    assert!(PolynomialMatrix::<Gf8B>::hermite_pade(&series, 2, &[1]).is_err());
}

#[test]
fn shift_and_column_rejections_cover_error_arms() {
    let input = PolynomialMatrix::from_entries(1, 1, alloc::vec![poly_b(&[b(1)])]).unwrap();
    let constraints = alloc::vec![InterpolationConstraint::new(b(0), 0, 1)];
    assert!(input.interpolation_basis(&constraints, &[]).is_err());
    assert!(
        input
            .interpolation_moduli(&[InterpolationConstraint::new(b(0), 3, 1)])
            .is_err()
    );
    // Zero-multiplicity constraints impose nothing: identity with zero count.
    let idle = alloc::vec![InterpolationConstraint::new(b(0), 0, 0)];
    let result = input.interpolation_basis(&idle, &[0]).unwrap();
    assert_eq!(result.basis, PolynomialMatrix::identity(1).unwrap());
    assert_eq!(result.independent_constraints, 0);
    // Empty series rejected across Padé constructors.
    assert!(PolynomialMatrix::<Gf8B>::simultaneous_pade(&[], 3, 1).is_err());
    // Formerly singular-block data under the old `[0, bound]` shifts now
    // selects under the balanced shifts: the Hankel determinant is
    // nonzero and the minimum-shifted-degree row carries the exact
    // denominator degree.
    let singular = poly_b(&[b(1), b(2), b(3)]);
    let relation = PolynomialMatrix::<Gf8B>::pade_relation(&singular, 3, 1).unwrap();
    assert_eq!(relation.denominator.degree(), Some(1));
    let product = singular.multiply(&relation.denominator).unwrap();
    let mut residual = relation.numerator.clone();
    residual.sub_assign(&product).unwrap();
    for coefficient in 0..3 {
        assert!(residual.coefficient(coefficient).is_zero());
    }
}

extern crate alloc;
