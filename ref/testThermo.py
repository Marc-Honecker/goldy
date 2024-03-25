import numpy as np

# mass = k_spring = temp = 1

mass = 2

d_time = 2 * np.pi / 80
gamma = 0.01
n_atom = 6000

n_relax = 4 * np.maximum(100, int(1 / gamma))
n_observ = 64 * n_relax
n_loop = 16

rand_force_pre = np.sqrt(6.0 * mass * gamma / d_time)
x = np.random.normal(0.0, 1.0, (n_atom, 3))
v = np.random.normal(0.0, 1.0, (n_atom, 3))

for i_loop in range(n_loop):
    t_kin_1 = t_kin_2 = 0
    v_pot_1 = v_pot_2 = 0
    for i_time in range(n_relax + n_observ):
        force = -x
        force -= gamma * v
        force += rand_force_pre * np.random.uniform(-1.0, 1.0, size=(n_atom, 3))
        force /= mass
        v += force * d_time
        x += v * d_time
        if i_time > n_relax:
            v_pot_mean = np.mean(np.sum(x**2, axis=1)) / 2
            t_kin_mean = mass * np.mean(np.sum(v**2, axis=1)) / 2
            v_pot_1 += v_pot_mean
            v_pot_2 += v_pot_mean**2
            t_kin_1 += t_kin_mean
            t_kin_2 += t_kin_mean**2
    v_pot_1 /= n_observ
    v_pot_2 /= n_observ
    t_kin_1 /= n_observ
    t_kin_2 /= n_observ
    print(v_pot_1, v_pot_2, t_kin_1, t_kin_2)
