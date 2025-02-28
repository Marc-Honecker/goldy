use assert_approx_eq::assert_approx_eq;
use goldy_core::force_update::ForceUpdateBuilder;
use goldy_core::gronbech_jensen::GronbechJensen;
use goldy_core::neighbor_list::NeighborList;
use goldy_core::observer::Observer;
use goldy_core::potential::pair_potential::PairPotential;
use goldy_core::potential::pair_potential_collection::PairPotentialCollectionBuilder;
use goldy_core::simulation_box::BoundaryTypes;
use goldy_core::storage::atom_type::AtomTypeBuilder;
use goldy_core::system::System;
use nalgebra::Vector3;

const DIM: usize = 3;

#[test]
#[ignore]
fn gj_test() {
    // simulation parameters
    let temp = 0.7;
    let runs = 2_500;
    let warm_up = 1_500;
    let gamma = 10.0;
    let mass = 1.0;
    let dt = 0.005;

    let num_runs = 1;
    let (mut mean_ke, mut mean_ke_sec) = (0.0, 0.0);
    let (mut mean_pe, mut mean_pe_sec) = (0.0, 0.0);

    for i in 0..num_runs {
        let (ke, ke_sec, pe, pe_sec) = run_gj(temp, runs, warm_up, gamma, mass, dt);
        println!("{i}\t{ke}\t{ke_sec}\t{pe}\t{pe_sec}");

        mean_ke += ke;
        mean_ke_sec += ke_sec;

        mean_pe += pe;
        mean_pe_sec += pe_sec;
    }

    mean_ke /= num_runs as f64;
    mean_ke_sec /= num_runs as f64;

    mean_pe /= num_runs as f64;
    mean_pe_sec /= num_runs as f64;

    println!("\n{mean_ke}\t{mean_ke_sec}\t{mean_pe}\t{mean_pe_sec}");

    // this should hold
    let analytical_solution = 1.5 * temp;
    assert_approx_eq!(analytical_solution, mean_ke, 1e-7 * analytical_solution);
}

fn run_gj(
    temp: f64,
    runs: usize,
    warm_up: usize,
    gamma: f64,
    mass: f64,
    dt: f64,
) -> (f64, f64, f64, f64) {
    let at = AtomTypeBuilder::default()
        .id(1)
        .damping(gamma)
        .mass(mass)
        .build()
        .unwrap();

    // the atoms
    let mut system: System<f64, DIM> =
        System::new_cubic(Vector3::from_element(8), 0.94, BoundaryTypes::Periodic, at);

    // creating the Lowest-Order Langevin integrator
    let mut gj = GronbechJensen::new(system.number_of_atoms());

    // lennard-jones
    let lj = PairPotential::new_lennard_jones(1.0, 2f64.powf(1.0 / 6.0), 3.5);
    let potential = PairPotentialCollectionBuilder::default()
        .add_potential(&at, &at, lj)
        .build()
        .unwrap();

    let mut updater = ForceUpdateBuilder::default()
        .potential(Box::new(potential.clone()))
        .build();

    // the neighbor list
    let mut neighbor_list = NeighborList::new(
        &system.atoms.x,
        &system.atoms.atom_types,
        &system.sim_box,
        &potential,
    );

    //
    let mut observer = Observer::new();

    // the main MD-loop
    for i in 0..runs {
        // setting the forces to zero
        system.atoms.f.set_to_zero();

        updater.update_forces(&mut system.atoms, &neighbor_list, &system.sim_box, temp, dt);

        // propagating the system in time
        gj.propagate(&mut system.atoms, dt, temp);

        system
            .sim_box
            .apply_boundary_conditions(&mut system.atoms.x);

        neighbor_list.update(
            &system.atoms.x,
            &system.atoms.atom_types,
            &system.sim_box,
            &potential,
        );

        if i >= warm_up && i % 100 == 0 {
            // measuring the kinetic energy
            observer.observe_kinetic_energy(&system.atoms);

            observer.observe_potential_energy(&system, &updater, &neighbor_list);
        }
    }

    (
        observer.get_mean_kinetic_energy(),
        observer.get_second_moment_kinetic_energy(),
        observer.get_mean_potential_energy().unwrap(),
        observer.get_second_moment_potential_energy().unwrap(),
    )
}
