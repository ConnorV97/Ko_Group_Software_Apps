import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, pi

a = 0.350

def rectangle(width, height):
    x0= width/2
    y0= height/2
    return pb.Polygon([[x0,y0], [x0,-y0], [-x0,-y0], [-x0,y0]])

def ldos_from_params_TriLat(pos_dopants, pos_vac, theta, shift_x, shift_y, l_STM, U, sig, E_range, E_reso, gamma):

    l_model =l_STM + 15

    a1= np.array([a,0])
    a2 = np.array([a/2, a/2*sqrt(3)])

    site_pos = np.array([0.0, 0.0])

    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    a1=np.matmul(rot_matrix,a1)
    a2=np.matmul(rot_matrix,a2)
    site_pos=np.matmul(rot_matrix,site_pos)
    shift = np.array([shift_x, shift_y])

    def triangular_lattice():

        t_nn= 0.92      # [eV] for nearest neighbor hopping
        t_nnn = 0.00    # [eV} for nnn hopping

        lat = pb.Lattice(a1, a2)
        lat.add_sublattices(('Pb', np.add(site_pos, shift))
                            )
        lat.add_hoppings(
            ([1, 0], 'Pb', 'Pb', t_nn),
            ([0, 1], 'Pb', 'Pb', t_nn),
            ([1, -1], 'Pb', 'Pb', t_nn),
        )
        return lat

    def dopants(pos_dopants, sigma, V):
        @pb.onsite_energy_modifier
        def potential(x,y):
            pot = 0
            for pos in pos_dopants:
                x0, y0 = pos
                pot+= np.exp(-0.5*((x-x0)**2 + (y-y0)**2)/sigma**2)
            return V * pot
        return potential

    def vacancy(position, radius):
        @pb.site_state_modifier
        def modifier(state, x, y):
            for pos in position:
                state[(x-pos[0])**2 + (y-pos[1])**2 < radius**2]= False
            return state
        return modifier

    has_dopants = len(pos_dopants) > 0
    has_vacancy = len(pos_vac) > 0

    if not has_dopants and not has_vacancy:
        model = pb.Model(
            triangular_lattice(),
            rectangle(l_model, l_model)
        )
    elif has_dopants and not has_vacancy:
        model = pb.Model(
            triangular_lattice(),
            rectangle(l_model, l_model),
            dopants(pos_dopants, sigma = sig, V=U)
        )
    elif not has_dopants and has_vacancy:
        model = pb.Model(
            triangular_lattice(),
            rectangle(l_model, l_model),
            vacancy(pos_vac, radius= 0.01)
        )
    else:
        model = pb.Model(
            triangular_lattice(),
            rectangle(l_model, l_model),
            dopants(pos_dopants, sigma = sig, V=U),
            vacancy(pos_vac, radius= 0.01)
        )

    kpm = pb.kpm(model)

    import scipy.sparse.linalg as spla

    H = model.hamiltonian
    print("Hamiltonian shape:", H.shape)
    print("Number of sites:", H.shape[0])
    print("Data type:", H.dtype)

    # Check for NaN or Inf in the Hamiltonian
    print("Any NaN:", np.any(np.isnan(H.data)))
    print("Any Inf:", np.any(np.isinf(H.data)))

    # Get the actual spectral radius
    try:
        k = min(2, H.shape[0] - 1)
        if k > 0:
            eigs = spla.eigsh(H, k=k, which='BE', return_eigenvectors=False)
            print("Eigenvalues:", eigs)
        else:
            print("Matrix too small for eigsh")

    except Exception as e:
        print("Eigensolver failed:", e)


    l_dos = l_STM + 1
    energies = np.linspace(-E_range, E_range, E_reso)

    spatial_ldos = kpm.calc_spatial_ldos(energy=energies, broadening= gamma, shape= rectangle(l_dos, l_dos))

    smap = spatial_ldos.structure_map(0)
    ldos_positions = np.transpose(np.array(smap.positions))
    num_points = ldos_positions.shape[0]
    ldos = np.zeros((energies.shape[0],num_points, 4))

    for i, energy in enumerate(energies):
        smap = spatial_ldos.structure_map(energy=energy)
        ldos_data = np.array(smap.data)
        ldos_positions = np.transpose(np.array(smap.positions))
        ldos[i,:,0:3]= ldos_positions[:,0:3]
        ldos[i, :,  3  ] = np.array(smap.data, ndmin=1)
        i+=1

    return ldos, model


def few_dopants_distri_TL(l_STM, num_dopants, num_vac, theta, shift_x, shift_y):
    """
    Randomly places dopants and vacancies on lattice sites of the Pb(111)
    triangular lattice that fall within the STM image area.

    Parameters
    ----------
    l_STM : float
        Lateral size of the STM image [nm].
    num_dopants : int
        Number of dopant atoms to place.
    num_vac : int
        Number of vacancies to place.
    theta : float
        Lattice rotation angle [radians].
    shift_x, shift_y : float
        Origin shift [nm].

    Returns
    -------
    pos_dopants : list of [x, y] arrays
    pos_vac     : list of [x, y] arrays
    """

    # Lattice vectors (before rotation) — same a as the LDOS solver
    a1 = np.array([a, 0.0])
    a2 = np.array([a / 2.0, a / 2.0 * sqrt(3)])

    # Rotation
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    a1 = np.matmul(rot_matrix, a1)
    a2 = np.matmul(rot_matrix, a2)
    shift = np.array([shift_x, shift_y])

    # n_max: enough unit cells to cover the image (same logic as MLG)
    n_max = int(np.round(l_STM * 2))

    pos_dopants = []
    pos_vac = []

    # ------------------------------------------------------------------
    # Place dopants
    # ------------------------------------------------------------------
    # Triangular lattice has one site per unit cell, so no sublattice
    # selection is needed — every (m, n) pair is a valid lattice site.
    for _ in range(int(num_dopants)):
        m = np.random.randint(-n_max, n_max + 1)
        n = np.random.randint(-n_max, n_max + 1)
        pos_ = m * a1 + n * a2  # position without shift (for boundary check)
        # Only keep if the site falls inside the STM image
        if ((-l_STM / 2 <= pos_[0] <= l_STM / 2) and
                (-l_STM / 2 <= pos_[1] <= l_STM / 2)):
            pos_dopants.append(m * a1 + n * a2 + shift)

    # ------------------------------------------------------------------
    # Place vacancies
    # ------------------------------------------------------------------
    for _ in range(int(num_vac)):
        m = np.random.randint(-n_max, n_max + 1)
        n = np.random.randint(-n_max, n_max + 1)
        pos_ = m * a1 + n * a2
        if ((-l_STM / 2 <= pos_[0] <= l_STM / 2) and
                (-l_STM / 2 <= pos_[1] <= l_STM / 2)):
            pos_vac.append(m * a1 + n * a2 + shift)

    return pos_dopants, pos_vac


