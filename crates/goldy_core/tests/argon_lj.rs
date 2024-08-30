use goldy_core::propagator::velocity_verlet::VelocityVerlet;
use goldy_core::simulation_box::SimulationBox;
use goldy_core::storage::vector::Positions;
use goldy_core::thermo::langevin::Langevin;
use goldy_core::Real;
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
    let runs = 10_000;
    let warm_ups = 2_000;
    let cutoff = 7.85723;

    // Argon
    let at = AtomTypeBuilder::default()
        .id(1)
        .damping(1e-2)
        .mass(39.95)
        .build()
        .unwrap();

    let lj = PairPotential::new_lennard_jones(120.0, 3.92862, cutoff);

    lj.write_to_file("test_outputs/lj.out");

    // Lennard-Jones pair-potential
    let pair_potential = PairPotentialCollectionBuilder::default()
        .add_potential(&at, &at, lj)
        .build()
        .unwrap();

    // the atoms
    let mut system = System::new_cubic(
        Vector3::new(7, 7, 7),
        0.5 * cutoff,
        BoundaryTypes::Periodic,
        at,
    );
    // let mut system = System::new_random(
    //     Vector3::new(120.0, 120.0, 120.0),
    //     BoundaryTypes::Periodic,
    //     at,
    //     400,
    // );

    // the neighbor-list
    let mut neighbor_list = compute_neighbor_list(&system.atoms.x, &system.sim_box, cutoff);

    // thermostat
    let langevin = Langevin::new();

    // creating the updater
    let mut updater = ForceUpdateBuilder::default()
        .thermostat(Box::new(langevin))
        .potential(Box::new(pair_potential))
        .build();

    // kinetic energy moments
    let mut tkin_1 = 0.0;
    let mut vpot_1 = 0.0;

    // the main MD-loop
    for i in 0..runs {
        if (i + 1) % 10 == 0 {
            // update the neighbor-list every 20 time-steps
            neighbor_list = compute_neighbor_list(&system.atoms.x, &system.sim_box, cutoff);
        }
        if i % 500 == 0 {
            // writing out the simulation cell
            system.write_system_to_file(format!("test_outputs/cubic_cell_{i}.out").as_str());
        }

        // propagating the system in time
        VelocityVerlet::integrate(
            &mut system.atoms,
            &neighbor_list,
            &system.sim_box,
            &mut updater,
            dt,
            temp,
        );

        if i > warm_ups {
            // measuring the kinetic energy
            // TODO: move to Observer
            let tkin_mean = 0.5
                * system
                    .atoms
                    .v
                    .iter()
                    .zip(&system.atoms.atom_types)
                    .map(|(&v, &t)| t.mass() * v.dot(&v))
                    .sum::<f32>();

            if i % 100 == 0 {
                let vpot_mean = updater
                    .measure_energy(
                        &system.atoms.x,
                        &neighbor_list,
                        &system.sim_box,
                        &system.atoms.atom_types,
                    )
                    .unwrap();

                vpot_1 += vpot_mean;

                println!(
                    "{i}, {}, {}",
                    tkin_mean / system.number_of_atoms() as f32,
                    vpot_mean / system.number_of_atoms() as f32
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

    tkin_1 /= ((runs - warm_ups) * system.number_of_atoms()) as f32;

    println!("{tkin_1}");

    // this should hold
    let analytical_solution = 1.5 * temp;
    assert_approx_eq!(analytical_solution, tkin_1, 1e-6 * analytical_solution);
}

fn compute_neighbor_list<T: Real>(
    x: &Positions<T, 3>,
    simulation_box: &SimulationBox<T, 3>,
    max_cutoff: T,
) -> Vec<Vec<usize>> {
    let mut neighbor_list = vec![Vec::new(); x.len()];
    let sq_max_cutoff = max_cutoff * max_cutoff;

    for (nl, x1) in neighbor_list.iter_mut().zip(x) {
        for (idx, x2) in x.iter().enumerate() {
            let sq_dist = simulation_box.sq_distance(x1, x2);

            if sq_dist <= sq_max_cutoff && sq_dist > T::zero() {
                nl.push(idx);
            }
        }
    }

    // println!("\n\nNeighbor List done!\n\n");

    neighbor_list
}
