use num_traits::Float;
use std::fs::File;
use std::io::Write;

use crate::Real;

// FIXME: Potential big bug!!!

const N: usize = 1024;

#[derive(Debug, Copy, Clone)]
pub struct PairPotential<T: Real> {
    sq_cutoff: T,
    dr: T,
    energies: [T; N + 1],
    pseudo_forces: [T; N + 1],
}

impl<T: Real> PairPotential<T> {
    /// Creates the 12-6 Mie potential.
    #[inline]
    pub fn new_lennard_jones(u0: T, r0: T, cutoff: T) -> Self {
        Self::new_mie(12, 6, u0, r0, cutoff)
    }

    /// Approximates the potential energy at a given squared distance `r_sq`.
    #[inline]
    pub fn energy(&self, r_sq: T) -> T {
        if r_sq > self.sq_cutoff {
            T::zero()
        } else {
            let (idx, weight) = self.get_idx(r_sq);

            weight * self.energies[idx] + (T::one() - weight) * self.energies[idx + 1]
        }
    }

    /// Approximates dU/dr^2 at a given squared distance `r_sq`.
    #[inline]
    pub fn pseudo_force(&self, r_sq: T) -> T {
        if r_sq > self.sq_cutoff {
            T::zero()
        } else {
            let (idx, weight) = self.get_idx(r_sq);

            weight * self.pseudo_forces[idx] + (T::one() - weight) * self.pseudo_forces[idx + 1]
        }
    }

    /// Dumps the `PairPotential` into the file.
    pub fn write_to_file(&self, filename: &str) {
        let mut file = File::create(filename).unwrap();

        let mut contents = String::new();

        self.energies
            .iter()
            .zip(&self.pseudo_forces)
            .enumerate()
            .for_each(|(idx, (energy, force))| {
                let r = Float::sqrt(if idx == 0 {
                    T::from(1e-4).unwrap()
                } else {
                    self.dr * T::from(idx).unwrap()
                });

                contents.push_str(format!("{r}\t{energy}\t{force}\n").as_str());
            });

        file.write_all(contents.as_bytes()).unwrap();
    }

    fn get_idx(&self, r_sq: T) -> (usize, T) {
        let idx = Float::trunc(r_sq / self.dr).to_usize().unwrap();
        let weight = T::one() - (r_sq - (T::from_usize(idx).unwrap() * self.dr)) / self.dr;

        (idx, weight)
    }

    #[inline]
    fn mie(u0: T, r0: T, n: i32, m: i32, r: T) -> T {
        let (f_n, f_m) = (T::from(n).unwrap(), T::from(m).unwrap());

        u0 / (f_n - f_m) * (f_m * Float::powi(r0 / r, n) - f_n * Float::powi(r0 / r, m))
    }

    #[inline]
    fn morse(u0: T, r0: T, n: i32, m: i32, r: T) -> T {
        let (n, m) = (T::from(n).unwrap(), T::from(m).unwrap());
        u0 / (n - m)
            * (m * Float::exp(n * (T::one() - r / r0)) - n * Float::exp(m * (T::one() - r / r0)))
    }
}

macro_rules! create_pair_potential {
    ($func_name: ident, $name: ident) => {
        impl<T: Real> PairPotential<T> {
            #[inline]
            pub fn $name(n: u32, m: u32, u0: T, r0: T, cutoff: T) -> Self {
                assert!(n > m, "Please provide proper exponents.");

                // creating the arrays
                let mut energies = [T::zero(); N + 1];
                let mut pseudo_forces = [T::zero(); N + 1];

                // computing the distance between two points.
                let dr = cutoff * cutoff / T::from(N).unwrap();
                // computing the energy at the cutoff distance
                let energy_at_cutoff = Self::$func_name(u0, r0, n as i32, m as i32, cutoff);

                energies
                    .iter_mut()
                    .zip(&mut pseudo_forces)
                    .enumerate()
                    .for_each(|(idx, (energy, force))| {
                        // computing the current distance
                        let r_sq = if idx == 0 {
                            // numerically stable
                            T::from(1e-4).unwrap()
                        } else {
                            dr * T::from(idx).unwrap()
                        };

                        // cut and shift the energy
                        *energy = Self::$func_name(u0, r0, n as i32, m as i32, Float::sqrt(r_sq))
                            - energy_at_cutoff;

                        // computing the pseudo force by numerical derivative.
                        let left = Self::$func_name(
                            u0,
                            r0,
                            n as i32,
                            m as i32,
                            Float::sqrt(r_sq * (T::one() - T::from(1e-5).unwrap())),
                        );
                        let right = Self::$func_name(
                            u0,
                            r0,
                            n as i32,
                            m as i32,
                            Float::sqrt(r_sq * (T::one() + T::from(1e-5).unwrap())),
                        );

                        *force = (right - left) / (T::from(1e-5).unwrap() * r_sq);
                    });

                Self {
                    sq_cutoff: Float::powi(cutoff, 2),
                    dr,
                    energies,
                    pseudo_forces,
                }
            }
        }
    };
}

create_pair_potential!(mie, new_mie);
create_pair_potential!(morse, new_morse);

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_get_idx() {
        // let's create a potential
        let potential = PairPotential {
            sq_cutoff: 10.0,
            dr: 1.0,
            energies: [0.0; N + 1],
            pseudo_forces: [0.0; N + 1],
        };

        // easy testcase
        // here idx has to be 10 and the weight 1.
        let (idx, weight) = potential.get_idx(10.0);
        assert_eq!(idx, 10);
        assert_approx_eq!(weight, 1f64);

        // a bit more advanced
        // here idx has to be 132 and the weight 0.7.
        let (idx, weight) = potential.get_idx(132.3);
        assert_eq!(idx, 132);
        assert_approx_eq!(weight, 0.7);
    }

    #[test]
    fn test_eval() {
        // setting the energies and pseudo forces up at exact points.
        let mut energies = [0.0; N + 1];
        let mut pseudo_forces = [0.0; N + 1];

        energies
            .iter_mut()
            .enumerate()
            .for_each(|(idx, x)| *x = idx as f64);
        pseudo_forces
            .iter_mut()
            .enumerate()
            .for_each(|(idx, x)| *x = idx as f64);

        // creating the PairPotential
        let pair_potential = PairPotential {
            sq_cutoff: 10.0,
            dr: 1.0,
            energies,
            pseudo_forces,
        };

        // testing the energy and pseudo force at r_sq = 10.0
        let e = pair_potential.energy(8.0);
        let f = pair_potential.pseudo_force(8.0);
        assert_approx_eq!(e, 8.0);
        assert_approx_eq!(f, 8.0);

        // testing the energy and pseudo force at r_sq = 4.5
        let e = pair_potential.energy(4.5);
        let f = pair_potential.pseudo_force(4.5);
        assert_approx_eq!(e, 4.5);
        assert_approx_eq!(f, 4.5);

        // testing the energy and pseudo force at r_sq > sq_cutoff
        let e = pair_potential.energy(12.0);
        let f = pair_potential.pseudo_force(12.0);
        assert_approx_eq!(e, 0.0);
        assert_approx_eq!(f, 0.0)
    }
}
