use std::{fs::OpenOptions, io::Write};

use goldy_core::{neighbor_list::NeighborList, storage::vector::Iterable};

#[test]
#[ignore]
fn ideal_gas() {
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::Vector3;

    use goldy_core::{
        force_update::ForceUpdateBuilder,
        propagator::{velocity_verlet::VelocityVerlet, Propagator},
        simulation_box::BoundaryTypes,
        storage::atom_type::AtomTypeBuilder,
        system::System,
        thermo::langevin::Langevin,
    };

    // simulation parameters
    let dt = 1.0e-2;
    let temp = 1.0;
    let runs = 8 * 8 * 8 * 10_000;
    let warm_up = 8 * 8 * 8 * 2_000;
    let gamma = 1.25e-3;

    // Argon
    let at = AtomTypeBuilder::default()
        .id(0)
        .damping(gamma)
        .mass(39.95)
        .build()
        .unwrap();

    // the atoms
    let mut system = System::new_cubic(Vector3::new(12, 12, 12), 3.0, BoundaryTypes::Open, at);

    // thermostat
    let langevin = Langevin::new();

    // creating the updater
    let mut updater = ForceUpdateBuilder::default()
        .thermostat(Box::new(langevin))
        .build();

    // kinetic energy moments
    let mut tkin_1 = 0.0;

    // the main MD-loop
    for i in 0..runs {
        // propagating the system in time
        VelocityVerlet::integrate(
            &mut system.atoms,
            &NeighborList::new_empty(),
            &system.sim_box,
            &mut updater,
            dt,
            temp,
        );

        if i >= warm_up {
            // measuring the kinetic energy
            let tkin_mean = 0.5
                * system
                    .atoms
                    .v
                    .iter()
                    .zip(&system.atoms.atom_types)
                    .map(|(&v, &t)| t.mass() * v.dot(&v))
                    .sum::<f64>();

            // updating the moments
            tkin_1 += tkin_mean;
        }
    }

    tkin_1 /= ((runs - warm_up) * system.number_of_atoms()) as f64;

    // creating the directory, if it does not exist
    std::fs::create_dir_all("test_outputs/thermostat_tests").unwrap_or(());

    // creating output file, if it doesn't exist yet and appending otherwise
    let mut kinetic_energy_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!(
            "test_outputs/thermostat_tests/langevin_{dt}_{gamma}.out"
        ))
        .expect("Could not open file");
    kinetic_energy_file
        .write(format!("{tkin_1}\n").as_bytes())
        .expect("writing failed");

    // this should hold
    let analytical_solution = 1.5 * temp;
    assert_approx_eq!(analytical_solution, tkin_1, 1e-3 * analytical_solution);
}
