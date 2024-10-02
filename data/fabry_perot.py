import numpy as np
import networkx as nx
import finesse
import pickle
import warnings
import h5py

import finesse.analysis.actions as fac
from finesse.analysis.actions import Xaxis, Series
from finesse.components.readout import ReadoutDetectorOutput

from utils import *

finesse.init_plotting(fmts=["png"])

NOMINAL_KATSCRIPT = """

    # Add a Laser named L0 with a power of 1 W.
    l l0 P=1
    bp roc_l0 l0.p1.o rc
    
    # Space attaching L0 <-> m1 with length of 0 m (default).
    s s0 l0.p1 m1.p1 0

    # Input mirror of cavity.
    m m1 R=0.9 T=0.1 Rc=-17

    # Intra-cavity space with length of 10 m.
    s LX m1.p2 m2.p1 L=10

    # End mirror of cavity.
    m m2 R=0.9 T=0.1 Rc=20

    cav cavity1 m1.p2.o priority=0

    noxaxis()
    """
def reset_model(kat):
    fabry_perot = finesse.Model()
    fabry_perot.parse(kat)
    fabry_perot.modes(maxtem=6, modes='even')
    return fabry_perot

if __name__ == '__main__':
    num_dps = 30000
    count = 0
    path = 'fabry_perot_files/'
    pct_pert = 0.0001
    
    fabry_perot = reset_model(NOMINAL_KATSCRIPT)
    
    finesse_g = fabry_perot.optical_network
    
    for node in finesse_g.nodes():
        name = node.replace('.', '_')
        fabry_perot.parse(f"pd p_{name} {node}")
        fabry_perot.parse(f"fd f_{name} {node} f=0")
        fabry_perot.parse(f"bp q_{name} {node} prop=q")
    data = []
    while count < num_dps:
        reset = False
        with warnings.catch_warnings():
            try:
                out = fabry_perot.run()
            except:
                reset = True
        if reset or not fabry_perot.cavity1.is_stable:
            # If we reached a point where the cavity is unstable, restart the random walk
            fabry_perot = reset_model(NOMINAL_KATSCRIPT)
            fabry_perot = perturb_model(fabry_perot, pct_pert)
            continue
        
        g = model_to_nx_port(fabry_perot)
        data.append(g)
        count+=1
        fabry_perot = perturb_model(fabry_perot, pct_pert)
        
        print(count)
        
    with h5py.File('fabry_perot_data.h5', 'w') as f:
        for i, graph in enumerate(data):
            serialized_graph = pickle.dumps(graph)

            # Store the serialized graph in the HDF5 file
            f.create_dataset(f'sim_{i}', data=np.void(serialized_graph))