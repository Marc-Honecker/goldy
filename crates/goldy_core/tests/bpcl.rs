use std::f64::consts::PI;

use assert_approx_eq::assert_approx_eq;
use goldy_core::observer::Observer;
use goldy_core::simulation_box::BoundaryTypes;
use goldy_core::storage::atom_type::AtomTypeBuilder;
use goldy_core::system::System;
use goldy_core::thermo::best_possible_conv_langevin::BestPossibleConvLangevin;
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

    let period = 2.0 * PI * gamma / 2.0f64.sqrt() * mass;
    let dt = period / 10.0;
    println!("{dt}, {period}, {}", dt / period);

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

    let sec_moment = observer.compute_nth_moment(&system.atoms.x, 2);
    let fourth_moment = observer.compute_nth_moment(&system.atoms.x, 4);

    println!("\n{}\n", fourth_moment / sec_moment.powi(2));

    // this should hold
    let analytical_solution = 1.5 * temp;
    assert_approx_eq!(
        analytical_solution,
        observer.get_mean_kinetic_energy(),
        1e-2 * analytical_solution
    );
}
