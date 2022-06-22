from qiskit_nature.drivers import PySCFDriver, UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
        ElectronicStructureDriverType,
        ElectronicStructureMoleculeDriver,
        )
from qiskit_nature.circuit.library import HartreeFock as HF
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.transformers import ActiveSpaceTransformer
#from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer
from qiskit_nature.operators.second_quantization import FermionicOp


# Get the inactive energy and the Hamiltonian operator in an active space
def GetHamiltonians(mol, n_orbs, na, nb):
    
    # construct the driver
    driver = PySCFDriver(molecule=mol, unit=UnitsType.ANGSTROM, basis='sto3g')

    # the active space transformer (use a (2, 2) active space)
    transformer = ActiveSpaceTransformer(num_electrons=(na+nb), num_molecular_orbitals=int(n_orbs/2))

    # the electronic structure problem
    problem = ElectronicStructureProblem(driver, [transformer])

    # get quantum molecule
    q_molecule = driver.run()

    # reduce the molecule to active space
    q_molecule_reduced = transformer.transform(q_molecule)

    # compute inactive energy
    core_energy = q_molecule_reduced.energy_shift["ActiveSpaceTransformer"]

    # add nuclear repulsion energy
    core_energy += q_molecule_reduced.nuclear_repulsion_energy

    ground_state = q_molecule_reduced.hf_energy

    # generate the second-quantized operators
    second_q_ops = problem.second_q_ops()

    # construct a qubit converter
    qubit_converter = QubitConverter(JordanWignerMapper())

    # qubit Operations 
    qubit_op = qubit_converter.convert(second_q_ops[0])

    # return the qubit operations
    return qubit_op, core_energy, ground_state

def getHamiltonian2(molecule):
    driver = ElectronicStructureMoleculeDriver(
            molecule,
            basis='sto6g',
            driver_type=ElectronicStructureDriverType.PYSCF
            )

    es_problem = ElectronicStructureProblem(driver)
    second_q_op = es_problem.second_q_ops()

    qubit_conversion = QubitConverter(mapper=JordanWignerMapper(), two_qubit_reduction=True)
    qubit_op = qubit_conversion.convert(second_q_op[0])

    return qubit_op


def main():
    molecule3 = Molecule(
            geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.735]]],
            charge=0,
            multiplicity=0,
            )

    molecule2 = Molecule(
            geometry=[["Na", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 1.8874]]],
            charge = 0,
            multiplicity=1
            )     

    molecule = Molecule(
            geometry=[["O", [0.0, 0.0, 0.1173]], ["H", [0.0, 0.7572, -0.4692]], ["H", [0.0, -0.7572, -0.4692]]],
            charge=0,
            )

    n_orbs = 8
    na = 2
    nb = 2

    q_op, core_e, ground_state = GetHamiltonians(molecule2, n_orbs, na, nb)

    print("Qubit Operation:")
    print(q_op)    

    print("Core Energy")
    print(core_e)
    
    print("Ground State Energy")
    print(ground_state)


 #   print("OG Method")
  #  q_op2 = getHamiltonian2(molecule)
   # print(q_op2)

    op_dict = {}
    for op in q_op:
        opstr = str(op).split()
        op_dict[opstr[2]] = float(opstr[0])
    
    # print(op_dict)
    #print(molecule.masses)

if __name__ == "__main__":
    main()


