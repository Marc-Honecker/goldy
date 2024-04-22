use std::f32::consts::PI;

use goldy_box::{BoundaryTypes, SimulationBoxBuilder};
use goldy_potential::{harmonic_oscillator::HarmonicOscillatorBuilder, Potential};
use goldy_storage::{
    atom_store::AtomStoreBuilder,
    atom_type::AtomTypeBuilder,
    atom_type_store::AtomTypeStoreBuilder,
    vector::{Forces, Positions, Velocities},
};
use goldy_thermo::{langevin::Langevin, ForceDrivenThermostat};
use nalgebra::Matrix3;

fn main() {
    // Let's spawn many atoms.
    let num_atoms = 10_000;

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
    let runs = 1_000_000;
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
        // initializing the forces
        // computing the Forces
        pot_energy += potential.eval(
            &atom_store.x,
            &mut atom_store.f,
            &sim_box,
            &atom_store.atom_types,
        );
        // adding non-deterministic forces
        langevin.thermo(
            &mut atom_store.f,
            &atom_store.v,
            &atom_store.atom_types,
            temp,
            dt,
        );

        // stepping forward in time
        atom_store
            .f
            .iter_mut()
            .zip(&atom_store.atom_types)
            .for_each(|(f, at)| *f /= at.mass());
        atom_store
            .v
            .iter_mut()
            .zip(&atom_store.f)
            .for_each(|(v, &f)| *v += f * dt);
        atom_store
            .x
            .iter_mut()
            .zip(&atom_store.v)
            .for_each(|(x, &v)| *x += v * dt);

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
