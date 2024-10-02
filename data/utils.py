import numpy as np
import finesse
import networkx as nx
import torch

def perturb_model(model, pct_pert):
    len_dofs = ['ls3', 'ls2', 'ls1', 'LX']
    for comp in model.elements:
        elem = getattr(model, comp)
        if isinstance(elem, finesse.components.mirror.Mirror):
            elem.Rc = elem.Rc*(1+(2*np.random.rand()-1)*pct_pert)
            elem.R = min(1, elem.R*(1+(2*np.random.rand()-1)*pct_pert))
            elem.T = 1-elem.R
        elif isinstance(elem, finesse.components.space.Space) and elem.name in len_dofs:
            elem.L =  elem.L.value*(1+(2*np.random.rand()-1)*pct_pert)
    return model

def model_to_nx_port(model):

    finesse_g = model.optical_network
    out = model.run()
    g = nx.DiGraph()
    
    for node in finesse_g.nodes():
        name = node.replace('.', '_')
        opt = node.split('.')[0]
        
        if isinstance(getattr(model, opt), finesse.components.mirror.Mirror):
            # Create feature vector
            g.add_node(node, Rc=getattr(model, opt).Rcx.value, R = getattr(model, opt).R.value, alpha=0, 
                       fd=torch.tensor(out[f'f_{name}'], dtype=torch.complex64), pd=out[f'p_{name}'], q=torch.tensor(out[f'q_{name}'], dtype=torch.complex64))
        elif isinstance(getattr(model, opt), finesse.components.laser.Laser):
            g.add_node(node, Rc=out['roc_l0'], R = 0, alpha=0, 
                       fd=torch.tensor(out[f'f_{name}'], dtype=torch.complex64), pd=out[f'p_{name}'], q=torch.tensor(out[f'q_{name}'], dtype=torch.complex64))
        elif isinstance(getattr(model, opt), finesse.components.beamsplitter.Beamsplitter):
            g.add_node(node, Rc=getattr(model, opt).Rcx.value, R = getattr(model, opt).R.value, alpha = getattr(model, opt).alpha.value, 
                       fd=torch.tensor(out[f'f_{name}'], dtype=torch.complex64), pd=out[f'p_{name}'], q=torch.tensor(out[f'q_{name}'], dtype=torch.complex64))
        elif isinstance(getattr(model, opt), finesse.components.lens.Lens):
            g.add_node(node, Rc=2*getattr(model, opt).f.value, R = 0, alpha = 0, 
                       fd=torch.tensor(out[f'f_{name}'], dtype=torch.complex64), pd=out[f'p_{name}'], q=torch.tensor(out[f'q_{name}'], dtype=torch.complex64))
    # Access edge attributes
    for i, edge in enumerate(finesse_g.edges().data()):
        dat = list(finesse_g.edges().data())[i][2]['owner']()
        if not isinstance(dat, finesse.components.space.Space):
            g.add_edge(str(edge[0]), str(edge[1]), length=0, nr=1)
        else:
            g.add_edge(str(edge[0]), str(edge[1]), length=dat.L.value if not isinstance(dat.L.value, finesse.symbols.Symbol) else dat.L.value.eval(), nr=dat.nr.value if not hasattr(dat.nr.value, 'eval') else dat.nr.value.eval())
    
    return g