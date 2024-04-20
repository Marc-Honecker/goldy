#[test]
fn ideal_gas() {
    use assert_approx_eq::assert_approx_eq;
    use goldy_storage::{
        atom_type_store::AtomTypeStoreBuilder,
        vector::{Forces, Positions, Velocities},
    };

    use goldy_storage::atom_type::AtomTypeBuilder;
    use goldy_thermo::langevin::Langevin;
    use goldy_thermo::ForceDrivenThermostat;

    // simulation parameters
    let dt = 0.01;
    let temp = 1.0;
    let runs = 100_000;

    // Argon
    let at = AtomTypeBuilder::default()
        .gamma(0.01)
        .mass(39.95)
        .build()
        .unwrap();

    // the atoms
    let num_atoms = 1_000;
    let mut x = Positions::<f32, 3>::new_gaussian(num_atoms, 0.0, 1.0);
    let mut v = Velocities::zeros(num_atoms);
    let types = AtomTypeStoreBuilder::default()
        .add_many(at, num_atoms)
        .build();

    // thermostat
    let mut langevin = Langevin::new();

    // kinetic energy moments
    let mut tkin_1 = 0.0;

    // the main MD-loop
    for _ in 0..runs {
        // initliazing the forces.
        let mut f = Forces::zeros(num_atoms);
        // computing the forces (for ideal gas)
        langevin.thermo(&mut f, &v, &types, temp, dt);

        // propagating in time
        //
        // converting the forces to accelerations
        f.iter_mut().zip(&types).for_each(|(f, t)| *f /= t.mass());
        // updating the velocities
        v.iter_mut().zip(&f).for_each(|(v, f)| *v += *f * dt);
        // updating the positions
        x.iter_mut().zip(&v).for_each(|(x, v)| *x += *v * dt);

        // measuring the kinetic energy
        let tkin_mean = 0.5
            * v.iter()
                .zip(&types)
                .map(|(&v, &t)| t.mass() * v.dot(&v))
                .sum::<f32>();

        // updating the moments
        tkin_1 += tkin_mean;
    }

    tkin_1 /= (runs * num_atoms) as f32;

    // this should hold
    assert_approx_eq!(1.5 * temp, tkin_1, 1e-2);
}
