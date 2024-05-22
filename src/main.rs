use nalgebra::Vector3;

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
            .mass(39.95)
            .damping(0.01)
            .build()
            .unwrap(),
    );

    // the md parameters
    let runs = 500_000;
    let dt = 0.01;
    let temp = 100.0;

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
    for _ in 0..runs {
        pot_energy +=
            VelocityVerlet::integrate(&mut system.atoms, &system.sim_box, &mut updater, dt, temp)
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

    // dumping the results
    println!(
        "Mean potenital energy: {}",
        pot_energy / (system.number_of_atoms() * runs) as f32
    );
    println!(
        "Mean kinetic energy: {}",
        kin_energy / (system.number_of_atoms() * runs) as f32 * 0.5
    );
}
