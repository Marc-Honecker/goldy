use assert_approx_eq::assert_approx_eq;
use goldy_core::observer::Observer;
use goldy_core::simulation_box::BoundaryTypes;
use goldy_core::storage::atom_type::AtomTypeBuilder;
use goldy_core::storage::vector::{Iterable, Positions};
use goldy_core::system::System;
use goldy_core::thermo::best_possible_conv_langevin::BestPossibleConvLangevin;
use goldy_core::Real;
use nalgebra::SVector;

const DIM: usize = 3;

#[test]
#[ignore]
fn bpcl_test() {
    // simulation parameters
    let temp = 1.0;
    let runs = 2_000;
    let warm_up = 1_000;
    let gamma = 1.0;
    let mass = 1.0;
    let dt = 0.01;

    let at = AtomTypeBuilder::default()
        .id(1)
        .damping(gamma)
        .mass(mass)
        .build()
        .unwrap();

    // the atoms
    let mut system: System<f64, DIM> =
        System::new_random(SVector::from_element(100.0), BoundaryTypes::Open, at, 2_000);

    // creating the Lowest-Order Langevin integrator
    let mut bpcl = BestPossibleConvLangevin::new();

    //
    let mut observer = Observer::new();

    // the main MD-loop
    for i in 0..runs {
        // propagating the system in time
        bpcl.propagate(&mut system.atoms, dt, temp);

        if i >= warm_up {
            // measuring the kinetic energy
            observer.observe_kinetic_energy(&system.atoms);
        }
    }

    let sec_moment = compute_nth_moment(&system.atoms.x, 2).sum() / DIM as f64;
    let fourth_moment = compute_nth_moment(&system.atoms.x, 4).sum() / DIM as f64;

    println!("\n{}\n", fourth_moment / sec_moment.powi(2));

    // this should hold
    let analytical_solution = 1.5 * temp;
    assert_approx_eq!(
        analytical_solution,
        observer.get_mean_kinetic_energy(),
        1e-2 * analytical_solution
    );
}

fn compute_mean<T: Real, const D: usize>(x: &Positions<T, D>) -> SVector<T, D> {
    x.iter().fold(SVector::zeros(), |acc, x| acc + x) / T::from(x.len()).unwrap()
}

fn compute_nth_moment<T: Real, const D: usize>(x: &Positions<T, D>, n: usize) -> SVector<T, D> {
    assert!(n > 1, "Please use compute_mean()");

    // computing the mean
    let mean = compute_mean(x);

    x.iter().fold(SVector::zeros(), |acc, x| {
        // |x - \mu|
        let mut diff = x - mean;
        // d_j = d_j^n
        diff.apply(|d| *d = num_traits::Float::powi(*d, n as i32));
        // sum over all |x - \mu|^n
        acc + diff
    }) / T::from(x.len()).unwrap()
}
