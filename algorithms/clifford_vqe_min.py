# Quantum chemistry example
# Developed with help from (OpenAI custom GPT) Elara
# (Requires OpenFermion)

from pyscf import gto, scf, ao2mo
from openfermion import MolecularData, FermionOperator, jordan_wigner, get_fermion_operator
from openfermionpyscf import run_pyscf

import multiprocessing
import numpy as np
import os


# Step 0: Set environment variables (before running script)

# On command line or by .env file, you can set the following variables
# QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1: For large circuits, automatically "elide," for approximation
# QRACK_NONCLIFFORD_ROUNDING_THRESHOLD=[0 to 1]: Sacrifices near-Clifford accuracy to reduce overhead
# QRACK_QUNIT_SEPARABILITY_THRESHOLD=[0 to 1]: Rounds to separable states more aggressively
# QRACK_QBDT_SEPARABILITY_THRESHOLD=[0 to 0.5]: Rounding for QBDD, if actually used

# Step 1: Define the molecule (Hydrogen, Helium, Lithium, Carbon, Nitrogen, Oxygen)

basis = "sto-3g"  # Minimal Basis Set
# basis = '6-31g'  # Larger basis set
# basis = 'cc-pVDZ' # Even larger basis set!
multiplicity = 1  # singlet, closed shell, all electrons are paired (neutral molecules with full valence)
# multiplicity = 2  # doublet, one unpaired electron (ex.: OH- radical)
# multiplicity = 3  # triplet, two unpaired electrons (ex.: O2)
charge = 0  # Excess +/- elementary charge, beyond multiplicity

# Hydrogen (and lighter):

# geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]  # H2 Molecule

# Helium (and lighter):

# geometry = [('He', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 7.74))]  # HeH Molecule

# Lithium (and lighter):

geometry = [('Li', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 15.9))]  # LiH Molecule

# Carbon (and lighter):

# Methane (CH4):
# geometry = [
#     ('C', (0.0000, 0.0000, 0.0000)),  # Central carbon
#     ('H', (1.0900, 0.0000, 0.0000)),  # Hydrogen 1
#     ('H', (-0.3630, 1.0270, 0.0000)),  # Hydrogen 2
#     ('H', (-0.3630, -0.5130, 0.8890)),  # Hydrogen 3
#     ('H', (-0.3630, -0.5130, -0.8890))  # Hydrogen 4
# ]

# Nitrogen (and lighter):

# geometry = [('N', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, 10.9))]  # N2 Molecule

# Ammonia:
# geometry = [
#     ('N', (0.0000, 0.0000, 0.0000)),  # Nitrogen at center
#     ('H', (0.9400, 0.0000, -0.3200)),  # Hydrogen 1
#     ('H', (-0.4700, 0.8130, -0.3200)), # Hydrogen 2
#     ('H', (-0.4700, -0.8130, -0.3200)) # Hydrogen 3
# ]

# Oxygen (and lighter):

# geometry = [('O', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 9.6))]  # OH- Radical
# geometry = [('O', (0.0000, 0.0000, 0.0000)), ('H', (0.7586, 0.0000, 0.5043)),  ('H', (-0.7586, 0.0000, 0.5043))]  # H2O Molecule
# geometry = [('C', (0.0000, 0.0000, 0.0000)), ('O', (0.0000, 0.0000, 1.128))]  # CO Molecule
# geometry = [('C', (0.0000, 0.0000, 0.0000)), ('O', (0.0000, 0.0000, 1.16)), ('O', (0.0000, 0.0000, -1.16))]  # CO2 Molecule
# geometry = [('O', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, 1.55))]  # NO Molecule
# geometry = [('O', (0.0, 0.0, 0.0)), ('N', (0.0, 0.0, 11.5))]  # NO+ Radical

# Nitrogen dioxide (toxic pollutant)
# geometry = [
#     ('N', (0.0000,  0.0000,  0.0000)),
#     ('O', (1.2000,  0.0000,  0.0000)),
#     ('O', (-1.2000 * np.cos(np.deg2rad(134)), 1.2000 * np.sin(np.deg2rad(134)), 0.0000))
# ]
# geometry = [('N', (0.0000, 0.0000, 0.0000)), ('N', (1.128, 0.0000, 0.0000)), ('O', (2.241, 0.0000, 0.0000))]  # N2O Molecule

# Ozone:
# geometry = [
#     ('O', (0.0000,  0.0000,  0.0000)),  # Central oxygen
#     ('O', (1.2780,  0.0000,  0.0000)),  # One oxygen
#     ('O', (-1.2780 * np.cos(np.deg2rad(117)), 1.2780 * np.sin(np.deg2rad(117)), 0.0000))  # The other oxygen
# ]

# H3O+ (Hydronium Ion):
# bond_length = 10.3  # Scaled bond length (1.03 Å in script units)
# bond_angle = 113.0  # Approximate bond angle of H3O+
# Convert angles to radians
# theta = np.deg2rad(bond_angle)
# Final geometry list
# geometry = [
#     ('O', (0.0, 0.0, 0.0)),
#     ('H', (bond_length, 0.0, 0.0)),
#     ('H', (-bond_length * np.cos(theta), bond_length * np.sin(theta), 0.0)),
#     ('H', (-bond_length * np.cos(theta), -bond_length * np.sin(theta), 0.0))
# ]

# Carbonate ion (CO3--):
# geometry = [
#     ('C', (0.0000, 0.0000, 0.0000)),  # Carbon at center
#     ('O', (1.2900, 0.0000, 0.0000)),  # Oxygen 1
#     ('O', (-0.6450, 1.1180, 0.0000)),  # Oxygen 2
#     ('O', (-0.6450, -1.1180, 0.0000))  # Oxygen 3
# ]

# Bicarbonate ion (HCO3-):
# geometry = [
#     ('C', (0.0000, 0.0000, 0.0000)),  # Carbon center
#     ('O', (1.2200, 0.0000, 0.0000)),  # Oxygen (C=O)
#     ('O', (-0.6100, 1.0550, 0.0000)),  # Oxygen (-O⁻)
#     ('O', (-0.6100, -1.0550, 0.0000)),  # Oxygen (OH)
#     ('H', (-1.2200, -1.0550, 0.0000))  # Hydrogen in OH group
# ]

# Sulfur chemistry:

# Hydrogen sulfide (rotten egg smell, major biologic sulfur compound):
# geometry = [
#     ('S', (0.0000, 0.0000, 0.0000)),
#     ('H', (1.3400, 0.0000, 0.0000)),
#     ('H', (-1.3400 * np.cos(np.deg2rad(92)), 1.3400 * np.sin(np.deg2rad(92)), 0.0000))
# ]

# Sulfur dioxide (major volcanic gas and pollutant):
# geometry = [
#     ('S', (0.0000, 0.0000, 0.0000)),  # Sulfur at center
#     ('O', (1.4300, 0.0000, 0.0000)),  # Oxygen 1
#     ('O', (-1.4300 * np.cos(np.deg2rad(119)), 1.4300 * np.sin(np.deg2rad(119)), 0.0000))  # Oxygen 2
# ]

# Sulfur trioxide (key in acid rain, forms H₂SO₄):
# geometry = [
#     ('S', (0.0000, 0.0000, 0.0000)),
#     ('O', (1.4200, 0.0000, 0.0000)),
#     ('O', (-0.7100, 1.2290, 0.0000)),
#     ('O', (-0.7100, -1.2290, 0.0000))
# ]

# Sulfate ion (SO4--, major oceanic anion, ionically bonds to Mg++):
# geometry = [
#     ('S', (0.0000, 0.0000, 0.0000)),
#     ('O', (1.4900, 0.0000, 0.0000)),
#     ('O', (-0.7450, 1.2900, 0.0000)),
#     ('O', (-0.7450, -1.2900, 0.0000)),
#     ('O', (0.0000, 0.0000, 1.4900))
# ]

# Oceanic electrolytes (consider isolating cations and anions as single atoms with excess charge):

# Sodium chloride:
# geometry = [
#     ('Na', (0.0000, 0.0000, 0.0000)),
#     ('Cl', (2.3600, 0.0000, 0.0000))
# ]

# Potassium chloride (biologically important):
# geometry = [
#     ('K', (0.0000, 0.0000, 0.0000)),
#     ('Cl', (2.6700, 0.0000, 0.0000))
# ]

# Calcium chloride:
geometry = [
    ('Ca', (0.0000, 0.0000, 0.0000)),
    ('Cl', (2.7800, 0.0000, 0.0000)),
    ('Cl', (-2.7800, 0.0000, 0.0000))
]

# Diatomic halogens:

# Fluorine gas (F2)
# geometry = [('F', (0.0000, 0.0000, 0.0000)), ('F', (1.4200, 0.0000, 0.0000))]

# Chlorine gas (Cl2)
# geometry = [('Cl', (0.0000, 0.0000, 0.0000)), ('Cl', (1.9900, 0.0000, 0.0000))]

# Silicon dioxide (quartz, sand, granite):
# geometry = [
#     ('Si', (0.0000, 0.0000, 0.0000)),  # Silicon at center
#     ('O', (1.6200, 0.0000, 0.0000)),  # Oxygen 1
#     ('O', (-1.6200, 0.0000, 0.0000))  # Oxygen 2
# ]

# The above are major atmospheric, oceanic, and soil components on Earth.
# Proper organic chemistry is beyond the scope of this script,
# but we give a memorable token example of a carbon ring.

# Benzene (C6H6)

# Define bond lengths (in angstroms, converted to script units)
# C_C = 13.9  # Carbon-carbon bond (1.39 Å)
# C_H = 10.9  # Carbon-hydrogen bond (1.09 Å)

# Angle of 120° between C-C bonds in the hexagonal ring
# theta = np.deg2rad(120)

# Define carbon positions (hexagonal ring)
# geometry = [
#     ('C', (C_C, 0.0, 0.0)),  # First carbon at x-axis
#     ('C', (C_C * np.cos(theta), C_C * np.sin(theta), 0.0)),
#     ('C', (-C_C * np.cos(theta), C_C * np.sin(theta), 0.0)),
#     ('C', (-C_C, 0.0, 0.0)),
#     ('C', (-C_C * np.cos(theta), -C_C * np.sin(theta), 0.0)),
#     ('C', (C_C * np.cos(theta), -C_C * np.sin(theta), 0.0))
# ]

# Define hydrogen positions (bonded to carbons)
# for i in range(6):
#     x, y, z = geometry[i][1]  # Get carbon position
#     hydrogen_x = x + (C_H * (x / C_C))  # Extend outward along C-C axis
#     hydrogen_y = y + (C_H * (y / C_C))
#     hydrogen_z = z  # Planar
#     geometry.append(('H', (hydrogen_x, hydrogen_y, hydrogen_z)))

# Now, `geometry` contains all 6 carbons and 6 hydrogens!

# Step 2: Compute the Molecular Hamiltonian
molecule = of.MolecularData(geometry, basis, multiplicity, charge)
molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
n_qubits = molecule.n_qubits  # Auto-detect qubit count
print(str(n_qubits) + " qubits...")

# Step 3: Convert to Qubit Hamiltonian (Jordan-Wigner)
def geometry_to_atom_str(geometry):
    """Convert list of (symbol, (x,y,z)) to Pyscf atom string."""
    return "; ".join(
        f"{symbol} {x:.10f} {y:.10f} {z:.10f}"
        for symbol, (x, y, z) in geometry
    )

# Convert and feed to gto.M()
atom_str = geometry_to_atom_str(geometry)

mol = gto.M(
    atom=atom_str,
    basis=basis
)

mf = scf.RHF(mol).run()

# Step 2: Create OpenFermion molecule
molecule_of = MolecularData(geometry, basis, multiplicity=1, charge=0)
molecule_of = run_pyscf(molecule_of, run_scf=True, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=False)
fermion_ham = get_fermion_operator(molecule_of.get_molecular_hamiltonian())
n_qubits = mol.nao << 1

# Step 3: Iterate JW terms without materializing full op
z_hamiltonian = []
    z_qubits = set()
    for paulis, coeff in hamiltonian.terms.items():
        # Skip if any X or Y in term
        if any(op in ("X", "Y") for _, op in paulis):
            continue

        q = []
        for qubit, op in paulis:
            # Z/I terms: keep only Z
            if op != "Z":
                continue
            q.append(qubit)
            z_qubits.add(qubit)

        z_hamiltonian.append((q, coeff))

z_hamiltonian = []
z_qubits = set()
for term, coeff in fermion_ham.terms.items():
    jw_term = jordan_wigner(FermionOperator(term=term, coefficient=coeff))  # Transform single term

    for pauli_string, jw_coeff in jw_term.terms.items():
        # Skip terms with X or Y
        if any(p in ('X', 'Y') for _, p in pauli_string):
            continue

        q = []
        for qubit, op in paulis:
            # Z/I terms: keep only Z
            if op != "Z":
                continue
            q.append(qubit)
            z_qubits.add(qubit)

        z_hamiltonian.append((q, coeff))

z_qubits = list(z_qubits)

# Step 4: Bootstrap!
def compute_energy(theta_bits, z_hamiltonian):
    """
    Computes the exact expectation value of a Hamiltonian on a computational basis state.

    Args:
        theta_bits: list of 0/1 integers representing the eigenstate in computational basis
        hamiltonian: list of (coefficient, PauliString) terms

    Returns:
        energy (float)
    """
    energy = 0.0

    for qubits, coeff in z_hamiltonian:
        # Z/I terms → product of ±1 from computational basis bits
        value = 1
        for qubit in qubits:
            if theta_bits[qubit] == 1:
                value *= -1
        energy += coeff * value

    return energy

# Parallelization by Elara (OpenAI custom GPT):
def bootstrap_worker(args):
    z_hamiltonian, theta, indices = args
    local_theta = theta.copy()
    flipped = []
    for i in indices:
        local_theta[i] = not local_theta[i]
        flipped.append(local_theta[i])
    energy = compute_energy(local_theta, z_hamiltonian)

    return indices, energy, flipped

def multiprocessing_bootstrap(z_hamiltonian, z_qubits):
    n_qubits = len(z_qubits)
    best_theta = np.random.randint(2, size=n_qubits)
    min_energy = compute_energy(best_theta, z_hamiltonian)
    iter_count = 0
    improved = True
    while improved:
        improved = False
        improved_2qb = True
        while improved_2qb:
            improved_2qb = False
            improved_1qb = True
            while improved_1qb:
                improved_1qb = False
                print(f"\nBootstrap Iteration {iter_count + 1}:")
                theta = best_theta.copy()

                with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                    args = []
                    for i in range(n_qubits):
                        args.append((z_hamiltonian, theta, (z_qubits[i],)))
                    results = pool.map(bootstrap_worker, args)

                results.sort(key=lambda r: r[1])
                indices, energy, flipped = results[0]
                if energy < min_energy:
                    min_energy = energy
                    for i in range(len(indices)):
                        best_theta[indices[i]] = flipped[i]
                    improved_1qb = True
                    print(f"  Qubit {indices[0]} flip accepted. New energy: {min_energy}")
                else:
                    print("  Qubit flips all rejected.")
                print(f"  {best_theta}")

                iter_count += 1

            if n_qubits < 2:
                break

            print(f"\nBootstrap Iteration {iter_count + 1}:")
            theta = best_theta.copy()

            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                args = []
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        args.append((z_hamiltonian, theta, (z_qubits[i], z_qubits[j])))
                results = pool.map(bootstrap_worker, args)

            results.sort(key=lambda r: r[1])
            indices, energy, flipped = results[0]
            if energy < min_energy:
                min_energy = energy
                for i in range(len(indices)):
                    best_theta[indices[i]] = flipped[i]
                improved_2qb = True
                print(f"  Qubits {indices} flip accepted. New energy: {min_energy}")
            else:
                print("  Qubit flips all rejected.")
            print(f"  {best_theta}")

            iter_count += 1

        if n_qubits < 3:
                break

        print(f"\nBootstrap Iteration {iter_count + 1}:")
        theta = best_theta.copy()

        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            args = []
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    for k in range(j + 1, n_qubits):
                        args.append((z_hamiltonian, theta, (z_qubits[i], z_qubits[j], z_qubits[k])))
            results = pool.map(bootstrap_worker, args)

        results.sort(key=lambda r: r[1])
        indices, energy, flipped = results[0]
        if energy < min_energy:
            min_energy = energy
            for i in range(len(indices)):
                best_theta[indices[i]] = flipped[i]
            improved = True
            print(f"  Qubit {indices} flip accepted. New energy: {min_energy}")
        else:
            print("  Qubit flips all rejected.")
        print(f"  {best_theta}")

        iter_count += 1

    return best_theta, min_energy

# Run threaded bootstrap
theta, min_energy = multiprocessing_bootstrap(z_hamiltonian, z_qubits)

print(f"\nFinal Bootstrap Ground State Energy: {min_energy} Ha")
print("Final Bootstrap Parameters:")
print(theta)
