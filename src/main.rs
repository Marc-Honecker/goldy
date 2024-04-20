use std::f32::consts::PI;

use goldy_box::{BoundaryTypes, SimulationBoxBuilder};
use goldy_potential::{harmonic_oscillator::HarmonicOscillatorBuilder, Potential};
use goldy_storage::{
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
    let mut x = Positions::<f32, 3>::new_gaussian(num_atoms, 0.0, 1.0);
    let mut v = Velocities::<f32, 3>::zeros(num_atoms);

    // the md parameters
    let runs = 100_000;
    let dt = 2.0 * PI / 80.0;
    let temp = 1.0;

    // The SimulationBox doesn't matter here, but
    // we still need to define it.
    let sim_box = SimulationBoxBuilder::default()
        .hmatrix(Matrix3::from_diagonal_element(10.0))
        .boundary_type(BoundaryTypes::Open)
        .build()
        .unwrap();
    let atom_types = AtomTypeStoreBuilder::default()
        .add_many(
            AtomTypeBuilder::default()
                .mass(1.0)
                .gamma(0.01)
                .build()
                .unwrap(),
            num_atoms,
        )
        .build();

    // defining the potential
    let potential = HarmonicOscillatorBuilder::default().k(1.0).build().unwrap();

    // defining the thermostat
    let mut langevin = Langevin::<f32>::new();

    // the potential energy
    let mut pot_energy = 0.0;
    // the kinetic energy
    let mut kin_energy = 0.0;

    // the main MD-loop
    for _ in 0..runs {
        // initializing the forces
        let mut f = Forces::<f32, 3>::zeros(num_atoms);
        // computing the Forces
        pot_energy += potential.eval(&x, &mut f, &sim_box, &atom_types);
        // adding non-deterministic forces
        langevin.thermo(&mut f, &v, &atom_types, temp, dt);

        // stepping forward in time
        f.iter_mut()
            .zip(&atom_types)
            .for_each(|(f, at)| *f /= at.mass());
        v.iter_mut().zip(&f).for_each(|(v, &f)| *v += f * dt);
        x.iter_mut().zip(&v).for_each(|(x, &v)| *x += v * dt);

        // updating kinetic energy
        kin_energy += v
            .iter()
            .zip(&atom_types)
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
