use std::fs::File;
use std::io::Write;

use goldy_core::force_update::ForceUpdateBuilder;
use goldy_core::neighbor_list::NeighborList;
use goldy_core::observer::Observer;
use goldy_core::potential::pair_potential::PairPotential;
use goldy_core::potential::pair_potential_collection::PairPotentialCollectionBuilder;
use goldy_core::propagator::Propagator;
use goldy_core::propagator::lowest_order_langevin::LowestOrderLangevin;
use goldy_core::simulation_box::BoundaryTypes;
use goldy_core::storage::atom_type::AtomTypeBuilder;
use goldy_core::system::System;
use nalgebra::Vector3;

const DIM: usize = 3;

#[test]
#[ignore]
fn lol_test() {
    // simulation parameters
    let temp = 0.67;
    let mut runs = 20_000;
    let mut warm_up = 15_000;
    let mut dt = 0.026;
    let gamma = 2.0;
    let mass = 1.0;
    let num_runs = 5;
    let num_observations = 12;

    // creating the directory, if it does not exist
    std::fs::create_dir_all("test_outputs").unwrap_or(());

    let mut output_file = File::create("test_outputs/lol.csv").unwrap();
    let header = "runs,warm_up,dt,ke,ke_sec,pe,pe_sec\n";
    output_file.write(header.as_bytes()).unwrap();

    for _ in 0..num_observations {
        let (mut mean_ke, mut mean_ke_sec) = (0.0, 0.0);
        let (mut mean_pe, mut mean_pe_sec) = (0.0, 0.0);

        for _ in 0..num_runs {
            let (ke, ke_sec, pe, pe_sec) = run_gj(temp, runs, warm_up, gamma, mass, dt);

            mean_ke += ke;
            mean_ke_sec += ke_sec;

            mean_pe += pe;
            mean_pe_sec += pe_sec;
        }

        mean_ke /= num_runs as f64;
        mean_ke_sec /= num_runs as f64;
        mean_pe /= num_runs as f64;
        mean_pe_sec /= num_runs as f64;

        let content = format!(
            "{runs},{warm_up},{dt:.5e},{mean_ke:.5e},{mean_ke_sec:.5e},{mean_pe:.5e},{mean_pe_sec:.5e}\n"
        );
        output_file.write_all(content.as_bytes()).unwrap();

        runs = runs * 5 / 4 + 500;
        warm_up = warm_up * 5 / 4 + 500;
        dt = dt * 0.8;
    }
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
    let mut system: System<f64, DIM> = System::new_cubic(
        Vector3::from_element(9),
        1.0817,
        BoundaryTypes::Periodic,
        at,
    );

    // creating the Lowest-Order Langevin integrator
    let mut lol = LowestOrderLangevin::new();

    // lennard-jones
    let lj = PairPotential::new_lennard_jones(1.0, 2f64.powf(1.0 / 6.0), 2.5);
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
        100,
    );

    // creating observer
    let mut observer = Observer::new();

    // the main MD-loop
    for i in 0..runs {
        lol.integrate(
            &mut system.atoms,
            &neighbor_list,
            &system.sim_box,
            &mut updater,
            dt,
            temp,
        );

        system
            .sim_box
            .apply_boundary_conditions(&mut system.atoms.x);

        neighbor_list.update(
            &system.atoms.x,
            &system.atoms.atom_types,
            &system.sim_box,
            &potential,
        );

        if i >= warm_up {
            // measuring the kinetic energy
            observer.observe_kinetic_energy(&system.atoms);

            observer.observe_potential_energy(&system, &mut updater, &neighbor_list);
        }
    }

    (
        observer.get_mean_kinetic_energy(),
        observer.get_second_moment_kinetic_energy(),
        observer.get_mean_potential_energy().unwrap(),
        observer.get_second_moment_potential_energy().unwrap(),
    )
}
