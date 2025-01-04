use assert_approx_eq::assert_approx_eq;
use nalgebra::Vector3;
use goldy_core::gronbech_jensen::GronbechJensen;
use goldy_core::simulation_box::BoundaryTypes;
use goldy_core::storage::atom_type::AtomTypeBuilder;
use goldy_core::system::System;

#[test]
#[ignore]
fn gronbech_jensen() {
    // simulation parameters
    let dt = 0.01;
    let temp = 1.0;
    let runs = 100_000;
    let warm_up = 20_000;

    // Argon
    let at = AtomTypeBuilder::default()
        .id(0)
        .damping(0.1)
        .mass(1.0)
        .build()
        .unwrap();

    // the atoms
    let mut system = System::new_cubic(Vector3::new(10, 10, 10), 2.0, BoundaryTypes::Open, at);

    // kinetic energy moments
    let mut tkin_1 = 0.0;

    // creating the Gronbech-Jensen integrator
    let mut gj = GronbechJensen::new(system.number_of_atoms());

    // the main MD-loop
    for i in 0..runs {
        // propagating the system in time
        gj.propagate(&mut system.atoms, dt, temp);

        if i >= warm_up {
            // measuring the kinetic energy
            let tkin_mean = 0.5
                * system
                    .atoms
                    .v
                    .iter()
                .zip(&system.atoms.atom_types)
                    .map(|(v, at)| v.norm_squared() * at.mass())
                    .sum::<f64>()
                    / system.number_of_atoms() as f64;

            tkin_1 += tkin_mean;
        }
    }

    tkin_1 /= ((runs - warm_up) * system.number_of_atoms()) as f64;

    // this should hold
    let analytical_solution = 1.5 * temp;
    assert_approx_eq!(analytical_solution, tkin_1, 1e-3 * analytical_solution);
}