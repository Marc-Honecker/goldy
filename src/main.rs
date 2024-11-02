use nalgebra::Vector3;

use goldy_core::neighbor_list::NeighborList;
use goldy_core::{
    force_update::ForceUpdateBuilder,
    potential::harmonic_oscillator::HarmonicOscillatorBuilder,
    propagator::{velocity_verlet::VelocityVerlet, Propagator},
    simulation_box::BoundaryTypes,
    storage::atom_type::AtomTypeBuilder,
    system::System,
    thermo::langevin::Langevin,
};

fn main() {
    // Let's spawn some atoms at cubic positions.
    let mut system = System::new_cubic(
        Vector3::new(10, 10, 10),
        5.0,
        BoundaryTypes::Periodic,
        AtomTypeBuilder::default()
            .id(0)
            .mass(1.0)
            .damping(0.01)
            .build()
            .unwrap(),
    );

    // the md parameters
    let runs = 2_000;
    let warm_up_runs = 600;
    let dt = 0.01;
    let temp = 1.0;

    // defining the potential
    let potential = HarmonicOscillatorBuilder::default().k(1.0).build().unwrap();

    // defining the thermostat
    let langevin = Langevin::new();

    let mut updater = ForceUpdateBuilder::default()
        .thermostat(Box::new(langevin))
        .potential(Box::new(potential))
        .build();

    // the potential energy
    let mut pot_energy = 0.0;
    // the kinetic energy
    let mut kin_energy = 0.0;

    // the main MD-loop
    for i in 0..runs {
        // stepping forward in time
        VelocityVerlet::integrate(
            &mut system.atoms,
            &NeighborList::new_empty(),
            &system.sim_box,
            &mut updater,
            dt,
            temp,
        );

        if i > warm_up_runs {
            // updating the potential energy
            pot_energy += updater
                .measure_energy(
                    &system.atoms.x,
                    &NeighborList::new_empty(),
                    &system.sim_box,
                    &system.atoms.atom_types,
                )
                .unwrap();

            // updating kinetic energy
            kin_energy += system
                .atoms
                .v
                .iter()
                .zip(&system.atoms.atom_types)
                .map(|(v, at)| v.dot(v) * at.mass())
                .sum::<f32>();
        }
    }

    // dumping the results
    println!(
        "Mean potential energy: {}",
        pot_energy / (system.number_of_atoms() * (runs - warm_up_runs)) as f32
    );
    println!(
        "Mean kinetic energy: {}",
        kin_energy / (system.number_of_atoms() * (runs - warm_up_runs)) as f32 * 0.5
    );
}
