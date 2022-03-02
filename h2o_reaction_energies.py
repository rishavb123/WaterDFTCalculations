from sparc.sparc_core import SPARC
from ase.build import molecule

# Initialize the data arrays
h_values = [0.2, 0.16, 0.14, 0.12]
h2o_values = []
h2_values = []
o2_values = []
reaction_values = []

# Loop over the h values
for h in h_values:
    # set up the calculator
    calc = SPARC(
        KPOINT_GRID=[1, 1, 1],
        h=h,
        EXCHANGE_CORRELATION="GGA_PBE",
        TOL_SCF=1e-5,
        RELAX_FLAG=1,
        PRINT_FORCES=1,
        PRINT_RELAXOUT=1,
    )

    # make the molecules
    water_molecule = molecule("H2O")
    hydrogen_molecule = molecule("H2")
    oxygen_molecule = molecule("O2")

    # set the calculator on the water molecule and calculate the potential energy
    water_molecule.set_calculator(calc)
    h2o_energy = water_molecule.get_potential_energy()
    h2o_values.append(h2o_energy)

    # set the calculator on the hydrogen molecule and calculate the potential energy
    hydrogen_molecule.set_calculator(calc)
    h2_energy = hydrogen_molecule.get_potential_energy()
    h2_values.append(h2_energy)

    # set the calculator on the oxygen molecule and calculate the potential energy
    oxygen_molecule.set_calculator(calc)
    o2_energy = oxygen_molecule.get_potential_energy()
    o2_values.append(o2_energy)

    # calculate the reaction energy
    reaction_energy = 2 * h2o_energy - 2 * h2_energy - o2_energy
    reaction_values.append(reaction_energy)

# save the results into a csv file
with open("reaction_energies.csv", "w") as f:
    f.write("h value,H2O,H2,O2,reaction\n")
    for h, h2o_energy, h2_energy, o2_energy, reaction_energy in zip(
        h_values, h2o_values, h2_values, o2_values, reaction_values
    ):
        f.write(
            "{},{},{},{},{}\n".format(
                h, h2o_energy, h2_energy, o2_energy, reaction_energy
            )
        )
