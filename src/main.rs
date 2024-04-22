use std::f32::consts::PI;

use goldy_box::{BoundaryTypes, SimulationBoxBuilder};
use goldy_potential::harmonic_oscillator::HarmonicOscillatorBuilder;
use goldy_propagator::{velocity_verlet::VelocityVerlet, Propagator};
use goldy_storage::{
    atom_store::AtomStoreBuilder,
    atom_type::AtomTypeBuilder,
    atom_type_store::AtomTypeStoreBuilder,
    vector::{Forces, Positions, Velocities},
};
use goldy_thermo::langevin::Langevin;
use nalgebra::Matrix3;

fn main() {
    // Let's spawn many atoms.
    let num_atoms = 1_000;

    // Let's spawn them at random positions.
    let x = Positions::<f32, 3>::new_gaussian(num_atoms, 0.0, 1.0);
    let v = Velocities::<f32, 3>::zeros(num_atoms);
    let f = Forces::<f32, 3>::zeros(num_atoms);
    let atom_types = AtomTypeStoreBuilder::default()
        .add_many(
            AtomTypeBuilder::default()
                .mass(1.0)
                .damping(0.01)
                .build()
                .unwrap(),
            num_atoms,
        )
        .build();

    let mut atom_store = AtomStoreBuilder::default()
        .positions(x)
        .velocities(v)
        .forces(f)
        .atom_types(atom_types)
        .build()
        .unwrap();

    // the md parameters
    let runs = 500_000;
    let dt = 2.0 * PI / 80.0;
    let temp = 1.0;

    // The SimulationBox doesn't matter here, but
    // we still need to define it.
    let sim_box = SimulationBoxBuilder::default()
        .hmatrix(Matrix3::from_diagonal_element(10.0))
        .boundary_type(BoundaryTypes::Open)
        .build()
        .unwrap();

    // defining the potential
    let potential = HarmonicOscillatorBuilder::default().k(1.0).build().unwrap();

    // defining the thermostat
    let mut langevin = Langevin::new();

    // the potential energy
    let mut pot_energy = 0.0;
    // the kinetic energy
    let mut kin_energy = 0.0;

    // the main MD-loop
    for _ in 0..runs {
        pot_energy += VelocityVerlet::integrate(
            &mut atom_store,
            &sim_box,
            Some(&potential),
            Some(&mut langevin),
            dt,
            temp,
        )
        .unwrap();

        // updating kinetic energy
        kin_energy += atom_store
            .v
            .iter()
            .zip(&atom_store.atom_types)
            .map(|(v, at)| v.dot(v) * at.mass())
            .sum::<f32>();
    }

    // dumping the results
    println!(
        "Mean potenital energy: {}",
        pot_energy / (num_atoms * runs) as f32
    );
    println!(
        "Mean kinetic energy: {}",
        kin_energy / (num_atoms * runs) as f32 * 0.5
    );
}
