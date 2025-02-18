use goldy_core::energy_observer::EnergyObserver;
use goldy_core::neighbor_list::NeighborList;
use goldy_core::propagator::leapfrog_verlet::LeapfrogVerlet;
use goldy_core::thermo::langevin::Langevin;
use nalgebra::Vector3;
use rand::Rng;

#[test]
#[ignore]
fn kob_andersen() {
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
    let runs = 20_000;
    let warm_ups = 10_000;
    let cutoff = 7.85723;

    // creating atomtype A
    let at_a = AtomTypeBuilder::default()
        .id(1)
        .damping(1e-2)
        .mass(1.0)
        .build()
        .unwrap();

    // creating atomtype B
    let at_b = AtomTypeBuilder::default()
        .id(2)
        .damping(1e-2)
        .mass(1.0)
        .build()
        .unwrap();

    // AA interaction: \epsilon = 1.0, \sigma = 1.0
    let lj_aa = PairPotential::new_lennard_jones(1.0 * 6.0, 1.0 * 2f32.powf(1.0 / 6.0), cutoff);
    // AB interaction: \epsilon = 1.5, \sigma = 0.8
    let lj_ab = PairPotential::new_lennard_jones(1.5 * 6.0, 0.8 * 2f32.powf(1.0 / 6.0), cutoff);
    // BB interaction: \epsilon = 0.5, \sigma = 0.88
    let lj_bb = PairPotential::new_lennard_jones(0.5 * 6.0, 0.88 * 2f32.powf(1.0 / 6.0), cutoff);

    // creating the directory, if it does not exist
    std::fs::create_dir_all("test_outputs").unwrap_or(());

    // Adding the potentials to the collection
    let pair_potential = PairPotentialCollectionBuilder::default()
        .add_potential(&at_a, &at_a, lj_aa)
        .add_potential(&at_a, &at_b, lj_ab)
        .add_potential(&at_b, &at_b, lj_bb)
        .build()
        .unwrap();

    // setting up the atoms by first setting all atomtypes to A
    let mut system =
        System::new_cubic(Vector3::new(12, 12, 12), 5.0, BoundaryTypes::Periodic, at_a);

    // and than changing 20% to B
    let mut rng = rand::thread_rng();
    system.atoms.atom_types.iter_mut().for_each(|at| {
        if rng.gen_range(0.0..1.0) > 0.8 {
            *at = at_b;
        }
    });

    system.write_system_to_file("test_outputs/kob_andersen0.out");

    // building the neighbor-list
    let mut neighbor_list = NeighborList::new(
        &system.atoms.x,
        &system.atoms.atom_types,
        &system.sim_box,
        &pair_potential,
    );

    // thermostat
    let langevin = Langevin::new();

    // creating the updater
    let mut updater = ForceUpdateBuilder::default()
        .thermostat(Box::new(langevin))
        .potential(Box::new(pair_potential.clone()))
        .build();

    // kinetic energy moments
    let (mut vpot_1, mut num_updates) = (0.0, 0);

    let mut observer = EnergyObserver::new();

    // the main MD-loop
    for i in 0..runs {
        // propagating the system in time
        LeapfrogVerlet::integrate(
            &mut system.atoms,
            &neighbor_list,
            &system.sim_box,
            &mut updater,
            dt,
            temp,
        );

        neighbor_list.update(
            &system.atoms.x,
            &system.atoms.atom_types,
            &system.sim_box,
            &pair_potential,
        );

        if i >= warm_ups {
            // measuring the kinetic energy
            observer.observe_kinetic_energy(&system.atoms);

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

                println!("{i}, {}", vpot_mean / system.number_of_atoms() as f32);
            }
        }

        // applying periodic boundary conditions
        system
            .sim_box
            .apply_boundary_conditions(&mut system.atoms.x);

        // after applying the boundary conditions, all atoms must still lie in it
        assert!(system.validate())
    }

    system.write_system_to_file(format!("test_outputs/kob_andersen{runs}.out").as_str());

    vpot_1 /= (num_updates * system.number_of_atoms()) as f32;

    println!("{vpot_1}");

    // this should hold
    let analytical_solution = 1.5 * temp;
    assert_approx_eq!(analytical_solution, observer.get_mean_kinetic_energy(), 0.5);
}
