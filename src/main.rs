use std::f64::consts::PI;

use nalgebra::Vector3;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rand_distr::{Distribution, StandardNormal, Uniform};

use goldy_core::{
    potential::{HarmonicOscillator, Potential},
    units::{Damping, Mass, Temperature, TimeStep},
    vector::{Accelaration, Force, Position, Velocity},
};

fn main() {
    let dt = TimeStep::new(2.0 * PI / 80.0).unwrap();
    let gamma = Damping::new(0.01, &dt).unwrap();
    let temp = Temperature::new(1.0).unwrap();
    let mass = Mass::new(1.0).unwrap();
    let num_atoms = 5_000;
    let k = 1.0;

    let num_relax_runs = 8 * 100.max((1.0 / *gamma) as u32);
    let num_observe_runs = 32 * num_relax_runs;
    let n_loop = 16;

    // initializing thermostat
    let mut langevin = Langevin::new(gamma, temp, mass);
    // initializing potential
    let mut potential = HarmonicOscillator::new(k);

    let mut rng = ChaChaRng::from_entropy();
    let normal = StandardNormal;
    let mut pos: Vec<_> = (0..num_atoms)
        .map(|_| Position::from_iterator(normal.sample_iter(&mut rng)))
        .collect();
    let mut vel: Vec<_> = (0..num_atoms)
        .map(|_| Velocity::from_iterator(normal.sample_iter(&mut rng)))
        .collect();

    for run in 0..n_loop {
        let (mut t_kin_1, mut t_kin_2) = (0.0, 0.0);
        let (mut v_pot_1, mut v_pot_2) = (0.0, 0.0);

        for i in 0..(num_relax_runs + num_observe_runs) {
            // computing -dU/dr
            let mut force = potential.force(&pos);

            // adding the langevin thermostat
            langevin.thermostat(&mut force, &vel);

            // converting forces to accelarations
            let acc: Vec<_> = force.iter().map(|f| f.to_accelaration(&mass)).collect();

            // propagating the system
            propagate(&mut pos, &mut vel, &acc, dt);

            if i > num_relax_runs {
                // calculating the mean kinetic energy
                let t_kin_mean = vel.iter().map(|vel| vel.norm_squared()).sum::<f64>()
                    / ((2 * num_atoms) as f64 * *mass);

                // calculating the mean potential energy
                let v_pot_mean = potential.energy(&pos).iter().fold(0.0, |acc, &e| acc + *e);

                // updating the potential energy moments
                v_pot_1 += v_pot_mean;
                v_pot_2 += v_pot_mean.powi(2);

                // updating the kinetic energy moments
                t_kin_1 += t_kin_mean;
                t_kin_2 += t_kin_mean.powi(2);
            }
        }

        // normalizing the potential energy moments
        v_pot_1 /= num_observe_runs as f64;
        v_pot_2 /= num_observe_runs as f64 * temp.powi(2);

        // normalizing the kinetic energy moments
        t_kin_1 /= num_observe_runs as f64;
        t_kin_2 /= num_observe_runs as f64 * temp.powi(2);

        // dumping the results
        println!("{run}\t{t_kin_1:.6}\t{t_kin_2:.6}\t{v_pot_1:.6}\t{v_pot_2:.6}");
    }
}

struct Langevin {
    rand_force_pre: f64,
    rng: ChaChaRng,
    uniform: Uniform<f64>,
    gamma: Damping,
}

impl Langevin {
    fn new(gamma: Damping, temp: Temperature, mass: Mass) -> Self {
        let rand_force_pre = (6.0 * *mass * *temp * *gamma).sqrt();
        let rng = ChaChaRng::from_entropy();
        let uniform = Uniform::new_inclusive(-1.0, 1.0);

        Self {
            rand_force_pre,
            rng,
            uniform,
            gamma,
        }
    }

    fn thermostat(&mut self, force: &mut [Force], vel: &[Velocity]) {
        force.iter_mut().zip(vel).for_each(|(force, vel)| {
            **force -= **vel * *self.gamma;
        });

        force.iter_mut().for_each(|force| {
            **force += Vector3::from_element(1.0)
                * self.rand_force_pre
                * self.uniform.sample(&mut self.rng);
        });
    }
}

#[inline]
fn propagate(pos: &mut [Position], vel: &mut [Velocity], acc: &[Accelaration], dt: TimeStep) {
    // propagating the velocities
    vel.iter_mut().zip(acc).for_each(|(vel, &acc)| {
        **vel += *acc * *dt;
    });

    // propagating the positions
    pos.iter_mut().zip(vel).for_each(|(pos, &mut vel)| {
        **pos += *vel * *dt;
    });
}
