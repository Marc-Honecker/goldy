use goldy_core::neighbor_list::NeighborList;
use goldy_core::observer::Observer;
use goldy_core::propagator::velocity_verlet::VelocityVerlet;
use goldy_core::rdf::RDF;
use goldy_core::thermo::langevin::Langevin;
use nalgebra::Vector3;

#[test]
#[ignore]
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
    let dt = 1e-3;
    let temp = 0.7;
    let runs = 40_000;
    let warm_ups = 20_000;
    let cutoff = 7.85723;

    // Argon
    let at = AtomTypeBuilder::default()
        .id(1)
        .damping(1e-2)
        .mass(39.95)
        .build()
        .unwrap();

    let u0 = 120.0;
    let r0 = 3.92862;

    let lj = PairPotential::new_lennard_jones(u0, r0, cutoff);

    // creating the directory, if it does not exist
    std::fs::create_dir_all("test_outputs").unwrap_or(());

    // Lennard-Jones pair-potential
    let pair_potential = PairPotentialCollectionBuilder::default()
        .add_potential(&at, &at, lj)
        .build()
        .unwrap();

    // the atoms
    let mut system = System::new_cubic(Vector3::new(12, 12, 12), 4.0, BoundaryTypes::Periodic, at);

    system.write_system_to_file("test_outputs/argon_0.out");

    // the neighbor-list
    let mut neighbor_list = NeighborList::new(
        &system.atoms.x,
        &system.atoms.atom_types,
        &system.sim_box,
        &pair_potential,
    );

    // thermostat
    let langevin = Langevin::new();

    let mut rdf = RDF::new(&at, &system.atoms, 1000, &pair_potential, 50.0);

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
        VelocityVerlet::integrate(
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

            if i % 100 == 0 {
                rdf.measure(&system.atoms, &neighbor_list, &system.sim_box);
            }

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

    system.write_system_to_file(format!("test_outputs/argon_{runs}.out").as_str());

    rdf.write("test_outputs/rdf.out");

    vpot_1 /= (num_updates * system.number_of_atoms()) as f32;

    println!("{vpot_1}");

    // this should hold
    let analytical_solution = 1.5 * temp;
    assert_approx_eq!(analytical_solution, observer.get_mean_kinetic_energy(), 0.5);
}
