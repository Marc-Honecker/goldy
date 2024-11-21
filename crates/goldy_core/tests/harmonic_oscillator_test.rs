use assert_approx_eq::assert_approx_eq;

use goldy_core::energy_observer::EnergyObserver;
use goldy_core::force_update::ForceUpdateBuilder;
use goldy_core::neighbor_list::NeighborList;
use goldy_core::potential::harmonic_oscillator::HarmonicOscillatorBuilder;
use goldy_core::propagator::leapfrog_verlet::LeapfrogVerlet;
use goldy_core::propagator::Propagator;
use goldy_core::simulation_box::BoundaryTypes;
use goldy_core::storage::atom_type::AtomTypeBuilder;
use goldy_core::system::System;
use goldy_core::thermo::langevin::Langevin;
use nalgebra::Vector3;

#[test]
fn harmonic_oscillator() {
    let (temp, dt) = (1.0, 1e-2);
    let (runs, warm_ups) = (5_000, 1_000);

    let at = AtomTypeBuilder::default()
        .id(1)
        .damping(1e-2)
        .mass(1.0)
        .build()
        .unwrap();

    let mut system =
        System::new_random(Vector3::new(50.0, 50.0, 50.0), BoundaryTypes::Open, at, 128);

    let potential = HarmonicOscillatorBuilder::default().k(1.0).build().unwrap();

    let thermostat = Langevin::new();

    let mut updater = ForceUpdateBuilder::default()
        .thermostat(Box::new(thermostat))
        .potential(Box::new(potential))
        .build();

    let mut observer = EnergyObserver::default();

    for i in 0..runs {
        LeapfrogVerlet::integrate(
            &mut system.atoms,
            &NeighborList::new_empty(),
            &system.sim_box,
            &mut updater,
            dt,
            temp,
        );

        system
            .sim_box
            .apply_boundary_conditions(&mut system.atoms.x);

        if i > warm_ups {
            observer.observe_kinetic_energy(&system.atoms);
        }
    }

    assert_approx_eq!(observer.get_mean_kinetic_energy(), 1.5f64 * temp, 5e-2);
}
