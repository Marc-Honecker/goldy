use std::f32::consts::PI;

use nalgebra::SVector;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rand_distr::{Distribution, StandardNormal, Uniform};

use goldy_core::{
    potential::{HarmonicOscillator, Potential},
    units::{Damping, Mass, Temperature, TimeStep},
    vector::{Accelaration, Force, Position, Velocity},
    Real,
};

fn main() {
    let dt = TimeStep::new(2.0 * PI / 80.0).unwrap();
    let gamma = Damping::new_raw(0.01).unwrap();
    let temp = Temperature::new(1.0).unwrap();
    let mass = Mass::new(1.0).unwrap();
    let num_atoms = 1_000;
    let k = 1.0;

    let num_relax_runs = 4 * 100.max((1.0 / (*gamma)) as usize);
    let num_observe_runs = 64 * num_relax_runs;
    let n_loop = 16;

    // initializing thermostat
    let mut langevin = Langevin::new(dt, gamma, temp, mass);
    // initializing potential
    let mut potential = HarmonicOscillator::new(k);

    // setting up random config
    let mut rng = ChaChaRng::from_entropy();
    let normal = StandardNormal;
    let mut pos: Vec<_> = (0..num_atoms)
        .map(|_| Position::<f32, 3>::from_iterator(normal.sample_iter(&mut rng)))
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
                let t_kin_mean = 0.5 * *mass * vel.iter().map(|vel| vel.dot(vel)).sum::<f32>()
                    / num_atoms as f32;

                // calculating the mean potential energy
                let v_pot_mean =
                    potential.energy(&pos).iter().fold(0.0, |acc, &e| acc + *e) / num_atoms as f32;

                // updating the potential energy moments
                v_pot_1 += v_pot_mean;
                v_pot_2 += v_pot_mean.powi(2);

                // updating the kinetic energy moments
                t_kin_1 += t_kin_mean;
                t_kin_2 += t_kin_mean.powi(2);
            }
        }

        // normalizing the potential energy moments
        v_pot_1 /= num_observe_runs as f32;
        v_pot_2 /= num_observe_runs as f32 * temp.powi(2);

        // normalizing the kinetic energy moments
        t_kin_1 /= num_observe_runs as f32;
        t_kin_2 /= num_observe_runs as f32 * temp.powi(2);

        // dumping the results
        println!("{run}\t{v_pot_1:.6}\t{v_pot_2:.6}\t{t_kin_1:.6}\t{t_kin_2:.6}");
    }
}

struct Langevin<T>
where
    T: Real + rand_distr::uniform::SampleUniform,
{
    rand_force_pre: T,
    rng: ChaChaRng,
    uniform: Uniform<T>,
    gamma: Damping<T>,
}

impl<T> Langevin<T>
where
    T: Real + rand_distr::uniform::SampleUniform,
{
    fn new(dt: TimeStep<T>, gamma: Damping<T>, temp: Temperature<T>, mass: Mass<T>) -> Self {
        let rand_force_pre =
            nalgebra::ComplexField::sqrt(T::from_f64(6.0).unwrap() * *mass * *temp * *gamma / *dt);
        let rng = ChaChaRng::from_entropy();
        let uniform =
            Uniform::<T>::new_inclusive(T::from_f64(-1.0).unwrap(), T::from_f64(1.0).unwrap());

        Self {
            rand_force_pre,
            rng,
            uniform,
            gamma,
        }
    }

    fn thermostat<const D: usize>(&mut self, force: &mut [Force<T, D>], vel: &[Velocity<T, D>]) {
        force.iter_mut().zip(vel).for_each(|(force, vel)| {
            **force -= **vel * *self.gamma;
        });

        force.iter_mut().for_each(|force| {
            **force += SVector::from_element(T::one())
                * self.rand_force_pre
                * self.uniform.sample(&mut self.rng);
        });
    }
}

#[inline]
fn propagate<T: Real, const D: usize>(
    pos: &mut [Position<T, D>],
    vel: &mut [Velocity<T, D>],
    acc: &[Accelaration<T, D>],
    dt: TimeStep<T>,
) {
    // propagating the velocities
    vel.iter_mut().zip(acc).for_each(|(vel, &acc)| {
        **vel += *acc * *dt;
    });

    // propagating the positions
    pos.iter_mut().zip(vel).for_each(|(pos, &mut vel)| {
        **pos += *vel * *dt;
    });
}
