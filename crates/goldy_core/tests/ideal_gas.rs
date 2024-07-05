#[test]
fn ideal_gas() {
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::Vector3;

    use goldy_core::{
        force_update::ForceUpdateBuilder,
        propagator::{Propagator, velocity_verlet::VelocityVerlet},
        simulation_box::BoundaryTypes,
        storage::atom_type::AtomTypeBuilder,
        system::System,
        thermo::langevin::Langevin,
    };

    // simulation parameters
    let dt = 0.01;
    let temp = 1.0;
    let runs = 500_000;

    // Argon
    let at = AtomTypeBuilder::default()
        .id(0)
        .damping(0.01)
        .mass(39.95)
        .build()
        .unwrap();

    // the atoms
    let mut system = System::new_cubic(Vector3::new(10, 10, 10), 2.0, BoundaryTypes::Open, at);

    // thermostat
    let langevin = Langevin::new();

    // creating the updater
    let mut updater = ForceUpdateBuilder::default()
        .thermostat(Box::new(langevin))
        .build();

    // kinetic energy moments
    let mut tkin_1 = 0.0;

    // the main MD-loop
    for _ in 0..runs {
        // propagating the system in time
        VelocityVerlet::integrate(
            &mut system.atoms,
            &Vec::new(),
            &system.sim_box,
            &mut updater,
            dt,
            temp,
        );

        // measuring the kinetic energy
        let tkin_mean = 0.5
            * system
                .atoms
                .v
                .iter()
                .zip(&system.atoms.atom_types)
                .map(|(&v, &t)| t.mass() * v.dot(&v))
                .sum::<f32>();

        // updating the moments
        tkin_1 += tkin_mean;
    }

    tkin_1 /= (runs * system.number_of_atoms()) as f32;

    // this should hold
    let analytical_solution = 1.5 * temp;
    assert_approx_eq!(analytical_solution, tkin_1, 1e-3 * analytical_solution);
}
