import os
import wizard.database
import wizard.calculator
from ase.io import read, write
from pynep.calculate import NEP
from wizard import plot_nep_2024
from wizard.io_generator import IOGenerator
from wizard.calculator import PropCalculator
from wizard.database import get_DB, get_ref_lattice








LIB = os.path.dirname(wizard.calculator.__file__)


models = ['/Users/wx/Desktop/Basic_properties']

FMAX = 0.02

SUPERCELL = (4,4,4)
CRYSTAL = 'fcc'
ELEMENTS = ['Au', 'Ni'] # NOTE: should be in the same sequence in nep.txt!! make sure its corresponding!!!
ISO_ENERGIES = {'Ni':-0.69369357, 'Au':-0.18439123}
# type_list = ['Truncated_Octahedron']
# format = ['core_shell', 'janus']
type_list = ['Regular_octahedron', 'Truncated_Octahedron', 'Cuboctahedron']
formats = ['core_shell', 'janus']

for potdir in models:
    BIMETALLIC = True
    # plot_nep_2024.plot_nep(potdir)

    for ELEMENT in ELEMENTS:

        # CASE 1
        # POT_NAME = ["pair_style eam/alloy",
        #             "pair_coeff * * {}/{}.eam.alloy {}".format(potdir, 'NiAu', ELEMENT) ]
        # FORMAT = 'lammps'

        # CASE 2
        POT_NAME = os.path.join(potdir, 'nep-NiAu.txt')
        FORMAT = 'nep'

        # CASE 3
        # POT_NAME = None
        # FORMAT = 'vasp'

        # LATTICE = get_ref_lattice(ELEMENT) # for VASP
        LATTICE = None # if None, then calculate lattice using BM fitting

        PC = PropCalculator(POT_NAME, FORMAT, LATTICE, potdir)
        IO = IOGenerator(POT_NAME, FORMAT, FMAX, potdir)
        

        # TODO: external interfaces!
        PC.calc_general_prop(IO, ELEMENT, CRYSTAL, SUPERCELL)
        PC.calc_vac_properties(IO, ELEMENT, CRYSTAL, SUPERCELL)
        PC.calc_cluster_properties(IO, ELEMENT, CRYSTAL, SUPERCELL, ISO_ENERGIES)
        if BIMETALLIC:
            PC.calc_bi_cluster_properties(IO, ELEMENTS, CRYSTAL, SUPERCELL, ISO_ENERGIES, type_list=type_list, formats=formats)
            BIMETALLIC = False
        else:
            print('Not do same.')




        


