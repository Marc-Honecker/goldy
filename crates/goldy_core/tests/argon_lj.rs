use goldy_core::compute_neighbor_list;
use goldy_core::propagator::leap_frog_verlet::LeapFrogVerlet;
use goldy_core::thermo::langevin::Langevin;
use nalgebra::Vector3;

#[test]
fn argon_lennard_jones() {
    use assert_approx_eq::assert_approx_eq;
    use goldy_core::{
        force_update::ForceUpdateBuilder,
        potential::{
            pair_potential::PairPotential,
            pair_potential_collection::PairPotentialCollectionBuilder,
        },
        propagator::Propagator,
        simulation_box::BoundaryTypes,
        storage::atom_type::AtomTypeBuilder,
        system::System,
    };

    // simulation parameters
    let dt = 1e-2;
    let temp = 1.0;
    let runs = 1_000_000;
    let warm_ups = 200_000;
    let cutoff = 7.85723;

    // Argon
    let at = AtomTypeBuilder::default()
        .id(1)
        .damping(1e-2)
        .mass(39.95)
        .build()
        .unwrap();

    let lj = PairPotential::new_lennard_jones(120.0, 3.92862, cutoff);

    // creating the directory, if it does not exist
    std::fs::create_dir_all("test_outputs").unwrap_or(());

    lj.write_to_file("test_outputs/lj.out");

    // Lennard-Jones pair-potential
    let pair_potential = PairPotentialCollectionBuilder::default()
        .add_potential(&at, &at, lj)
        .build()
        .unwrap();

    // the atoms
    let mut system = System::new_cubic(
        Vector3::new(4, 4, 4),
        0.45 * cutoff,
        BoundaryTypes::Periodic,
        at,
    );
    // let mut system = System::new_random(
    //     Vector3::new(4.5 * cutoff, 4.5 * cutoff, 4.5 * cutoff),
    //     BoundaryTypes::Periodic,
    //     at,
    //     864,
    // );

    // the neighbor-list
    let mut neighbor_list = compute_neighbor_list(&system.atoms.x, &system.sim_box, 1.1 * cutoff);

    // thermostat
    let langevin = Langevin::new();

    // creating the updater
    let mut updater = ForceUpdateBuilder::default()
        .thermostat(Box::new(langevin))
        .potential(Box::new(pair_potential))
        .build();

    // kinetic energy moments
    let mut tkin_1 = 0.0;
    let (mut vpot_1, mut num_updates) = (0.0, 0);

    // determines, when we need an update in the neighbor list
    let mut need_update = 2;

    // the main MD-loop
    for i in 0..runs {
        if (i + 1) % 4_000 == 0 && need_update < 600 {
            need_update *= 2;
        }

        if (i + 1) % need_update == 0 {
            // update neighbor list every few time steps
            neighbor_list = compute_neighbor_list(&system.atoms.x, &system.sim_box, 1.1 * cutoff);
        }

        if i % 10_000 == 0 {
            // writing out the simulation cell
            system.write_system_to_file(format!("test_outputs/cubic_cell_{i}.out").as_str());
        }

        // propagating the system in time
        LeapFrogVerlet::integrate(
            &mut system.atoms,
            &neighbor_list,
            &system.sim_box,
            &mut updater,
            dt,
            temp,
        );

        if i >= warm_ups {
            // measuring the kinetic energy
            // TODO: move to Observer
            let tkin_mean = 0.5
                * system
                    .atoms
                    .v
                    .iter()
                    .zip(&system.atoms.atom_types)
                    .map(|(&v, &t)| t.mass() * v.dot(&v))
                    .sum::<f64>();

            if i % 1_000 == 0 {
                let vpot_mean = updater
                    .measure_energy(
                        &system.atoms.x,
                        &neighbor_list,
                        &system.sim_box,
                        &system.atoms.atom_types,
                    )
                    .unwrap();

                vpot_1 += vpot_mean;
                num_updates += 1;

                println!(
                    "{i}, {}, {}",
                    tkin_mean / system.number_of_atoms() as f64,
                    vpot_mean / system.number_of_atoms() as f64
                );
            }

            // updating the moments
            tkin_1 += tkin_mean;
        }

        // applying periodic boundary conditions
        system
            .sim_box
            .apply_boundary_conditions(&mut system.atoms.x);

        // after applying the boundary conditions, all atoms must still lie in it
        assert!(system.validate())
    }

    tkin_1 /= ((runs - warm_ups) * system.number_of_atoms()) as f64;
    vpot_1 /= (num_updates * system.number_of_atoms()) as f64;

    println!("{tkin_1}, {vpot_1}");

    // this should hold
    let analytical_solution = 1.5 * temp;
    assert_approx_eq!(analytical_solution, tkin_1, 1e-2 * analytical_solution);
}
