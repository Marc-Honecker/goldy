use std::fs::File;
use std::io::Write;

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
    let temp = 0.67;
    let mut runs = 6_000;
    let mut warm_up = 5_000;
    let gamma = 10.0;
    let mass = 1.0;
    let num_runs = 1;

    let dts = [
        0.0174533,
        //0.0138636,
        //0.0110123,
        //0.00874737,
        //0.00694828,
        //0.00551922,
        //0.00438407,
        //0.00348239,
        //0.00276616,
        //0.00219724,
        //0.00174533,
        //0.00138636,
        //0.00110123,
        //0.000874737,
    ];

    // creating the directory, if it does not exist
    std::fs::create_dir_all("test_outputs").unwrap_or(());

    let mut output_file = File::create("test_outputs/gj.out").unwrap();
    let header = "runs,warm_up,dt,ke,ke_sec,pe,pe_sec\n";
    output_file.write(header.as_bytes()).unwrap();

    for dt in dts {
        let (mut mean_ke, mut mean_ke_sec) = (0.0, 0.0);
        let (mut mean_pe, mut mean_pe_sec) = (0.0, 0.0);

        for _ in 0..num_runs {
            let (ke, ke_sec, pe, pe_sec) = run_gj(temp, runs, warm_up, gamma, mass, dt);

            mean_ke += ke;
            mean_ke_sec += ke_sec;

            mean_pe += pe;
            mean_pe_sec += pe_sec;
        }

        mean_ke /= num_runs as f32;
        mean_ke_sec /= num_runs as f32;
        mean_pe /= num_runs as f32;
        mean_pe_sec /= num_runs as f32;

        let content = format!(
            "{runs},{warm_up},{dt:.5e},{mean_ke:.5e},{mean_ke_sec:.5e},{mean_pe:.5e},{mean_pe_sec:.5e}\n"
        );
        output_file.write_all(content.as_bytes()).unwrap();

        runs = runs * 5 / 4;
        warm_up = warm_up * 5 / 4;
    }
}

fn run_gj(
    temp: f32,
    runs: usize,
    warm_up: usize,
    gamma: f32,
    mass: f32,
    dt: f32,
) -> (f32, f32, f32, f32) {
    let at = AtomTypeBuilder::default()
        .id(1)
        .damping(gamma)
        .mass(mass)
        .build()
        .unwrap();

    // the atoms
    let mut system: System<f32, DIM> = System::new_cubic(
        Vector3::from_element(9),
        1.0817,
        BoundaryTypes::Periodic,
        at,
    );

    // creating the Lowest-Order Langevin integrator
    let mut gj = GronbechJensen::new(system.number_of_atoms());

    // lennard-jones
    let lj = PairPotential::new_lennard_jones(1.0, 2f32.powf(1.0 / 6.0), 2.5);
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
