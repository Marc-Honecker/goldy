use goldy_core::neighbor_list::NeighborList;
use goldy_core::observer::Observer;
use goldy_core::propagator::leapfrog_verlet::LeapfrogVerlet;
use goldy_core::rdf::RDF;
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
    let dt = 5e-3;
    let temp = 0.466;
    let runs = 5_000;
    let warm_ups = 4_000;

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
    let (epsilon_aa, sigma_aa) = (1.0, 1.0);
    // AB interaction: \epsilon = 1.5, \sigma = 0.8
    let (epsilon_ab, sigma_ab) = (1.5, 0.8);
    // BB interaction: \epsilon = 0.5, \sigma = 0.88
    let (epsilon_bb, sigma_bb) = (0.5, 0.88);

    let lj_aa = PairPotential::new_lennard_jones(
        epsilon_aa,
        sigma_aa * 2f64.powf(1.0 / 6.0),
        2.5 * sigma_aa,
    );
    let lj_ab = PairPotential::new_lennard_jones(
        epsilon_ab,
        sigma_ab * 2f64.powf(1.0 / 6.0),
        2.5 * sigma_ab,
    );
    let lj_bb = PairPotential::new_lennard_jones(
        epsilon_bb,
        sigma_bb * 2f64.powf(1.0 / 6.0),
        2.5 * sigma_bb,
    );

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
    let mut system = System::new_cubic(
        Vector3::new(9, 9, 9),
        9.4 / 10.0,
        BoundaryTypes::Periodic,
        at_a,
    );

    // creating propagator
    let mut lfv = LeapfrogVerlet::new();

    // and than changing 20% to B
    let mut rng = rand::rng();
    system.atoms.atom_types.iter_mut().for_each(|at| {
        if rng.random_range(0.0..1.0) > 0.8 {
            *at = at_b;
        }
    });

    let mut rdf_aa = RDF::new(&at_a, &system.atoms, 500, &pair_potential, 50, 4.5);
    let mut rdf_bb = RDF::new(&at_b, &system.atoms, 500, &pair_potential, 50, 4.5);

    system.write_system_to_file("test_outputs/kob_andersen0.out");

    // building the neighbor-list
    let mut neighbor_list = NeighborList::new(
        &system.atoms.x,
        &system.atoms.atom_types,
        &system.sim_box,
        &pair_potential,
        100,
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

    let mut observer = Observer::new();

    // the main MD-loop
    for i in 0..runs {
        // propagating the system in time
        lfv.integrate(
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

            rdf_aa.measure(&system.atoms, &neighbor_list, &system.sim_box);
            rdf_bb.measure(&system.atoms, &neighbor_list, &system.sim_box);

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
                num_updates += 1;

                println!("{i}, {}", vpot_mean / system.number_of_atoms() as f64);
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

    vpot_1 /= (num_updates * system.number_of_atoms()) as f64;

    println!("{vpot_1}");

    rdf_aa.write("test_outputs/rdf_aa.csv");
    rdf_bb.write("test_outputs/rdf_bb.csv");

    // this should hold
    let analytical_solution = 1.5 * temp;
    assert_approx_eq!(
        analytical_solution,
        observer.get_mean_kinetic_energy(),
        3.0e-2
    );
}
