import numpy as np

def calc_surface(surface):
	mp_surfaces_energy_pred = np.array([mace_mp_l.get_potential_energy(struct) for struct in surface])
	np.save("mp_surfaces_energy_pred.npy", mp_surfaces_energy_pred)

