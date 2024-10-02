import numpy as np
import networkx as nx
import finesse
import pickle
import warnings
import h5py

import finesse.analysis.actions as fac
from finesse.analysis.actions import Xaxis, Series
from finesse.components.readout import ReadoutDetectorOutput

from sympy import symbols, solveset, limit

from utils import *

finesse.init_plotting(fmts=["png"])

NOMINAL_KATSCRIPT = """
    # modulators for core interferometer sensing - Advanced LIGO, CQG, 2015
    # http://iopscience.iop.org/article/10.1088/0264-9381/32/7/074001/meta#cqg507871s4-8
    variable f1 9099471
    variable f2 5*f1
    variable nsilica 1.45
    variable Xloss 60u
    variable Yloss 60u

    ###############################################################################
    ###   length definitions
    ###############################################################################
    variable Larm 3994.47
    variable LSR23 15.443  # distance between SR2 and SR3
    variable LSR3BS 19.366 # distance between SR3 and BS
    variable lmich 5.342   # average length of MICH
    variable lschnupp 0.08 # double pass schnupp length
    variable lSRC (17)*c0/(2*f2) # T1000298 Eq2.2, M=3

    ###############################################################################
    ###   laser
    ###############################################################################
    laser L0 P=125
    s s0 L0.p1 BS.p1
          
    ###############################################################################
    ###   BS
    ###############################################################################
    bs BS R=0.5 L=0 alpha=45 # Beam-splitter is 100% reflective since only one arm
    s BSsub1 BS.p3 BSAR1.p1 L=60m*cos(radians(29.186885954108114)) nr=nsilica
    s BSsub2 BS.p4 BSAR2.p2 L=60m*cos(radians(29.186885954108114)) nr=nsilica
    bs BSAR1 L=50u R=0 alpha=29.186885954108114
    bs BSAR2 L=50u R=0 alpha=29.186885954108114

    ###############################################################################
    ###   Xarm
    ###############################################################################
    # Distance from beam splitter to X arm input mirror
    s lx1 BSAR1.p3 ITMXlens.p1 L=lmich+lschnupp/2-ITMXsub.L*ITMXsub.nr-BSsub1.L*BSsub1.nr
    lens ITMXlens f=34500
    s lx2 ITMXlens.p2 ITMXAR.p1
    m ITMXAR R=0 L=0 xbeta=ITMX.xbeta ybeta=ITMX.ybeta phi=ITMX.phi
    s ITMXsub ITMXAR.p2 ITMX.p1 L=0.2 nr=nsilica
    m ITMX R=0.86 T=1-ITMX.R Rc=-1934
    s LX ITMX.p2 ETMX.p1 L=Larm
    m ETMX R=0.999995 T=1-ETMX.R Rc=2245
    cav cavXARM ETMX.p1.o

    ###############################################################################
    ###   SRC
    ###############################################################################
    s ls3 BSAR2.p4 SR3.p1 L=LSR3BS
    bs SR3 R=1 L=0 alpha=0.785 Rc=35.972841
    s ls2 SR3.p2 SR2.p1 L=LSR23
    bs SR2 R=1 L=0 alpha=-0.87 Rc=-6.406
    s ls1 SR2.p2 SRM.p1 L=lSRC-LSR3BS-LSR23-BSsub2.L*BSsub2.nr-lmich
    m SRM R=0.68 L=0 Rc=-5.6938
    s SRMsub SRM.p2 SRMAR.p1 L=0.0749 nr=nsilica
    m SRMAR R=0 L=0
                    
    ###############################################################################
    ### Length sensing and control
    ###############################################################################
    dof XARM ETMX.dofs.z
    dof CARM ETMX.dofs.z +1 # Different tham XARM?
    dof SRCL SRM.dofs.z +1 DC=90
    dof FRQ L0.dofs.frq
    dof RIN L0.dofs.amp
                    
    readout_rf rd_sr_f1 ETMX.p1.o f=f1 phase=0
    readout_rf rd_sr_f2 ETMX.p1.o f=f2 phase=0

    lock SRCL_lock rd_sr_f2.outputs.I SRCL.DC 1 1e-6
    lock CARM_lock rd_sr_f1.outputs.I CARM.DC -0.1 1e-6

    ###############################################################################
    ### DC power measurements
    ###############################################################################
    pd Pcirc ITMX.p2.i
    pd Psrc SRM.p1.i
    pd Pin ITMX.p1.i
    pd Pout SRMAR.p2.o

    pd Px ITMX.p2.o

    ad a_carrier_src SRM.p1.i f=0
    ad a_carrier_pinx ITMX.p1.i f=0
    ad a_carrier_x ITMX.p2.i f=0
    ad a_carrier_src_out_00 SRMAR.p2.o f=0 n=0 m=0
    ad a_carrier_src_out SRMAR.p2.o f=0
                    
    bp beam_p1_i ITMX.p1.i prop="q" direction=y q_as_bp=True
    bp beam_p2_o ITMX.p2.o prop="q" direction=y q_as_bp=True
    
    bp roc_l0 L0.p1.o prop='rc'

    mathd Pinx_carrier abs(a_carrier_pinx)**2
    mathd Pas_carrier abs(a_carrier_src)**2

    cav cavSRX SRM.p1.o via=ITMX.p1.i
    """

# Helper functions for locking the ALIGO IFO
def set_lock_gains(model, locks=None, d_dof=1e-9, gain_scale=1, verbose=False):
    """For the current state of the model each lock will have its gain computed. This is
    done by computing the gradient of the error signal with respect to the set feedback.

    The optical gain is then computed as -1/(slope).

    This function alters the state of the provided model.

    Parameters
    ----------
    model : Model
        Model to set the lock gains of
    locks : list
        Names of locks to set gains for, `None` is all enabled
    d_dof : double
        step size for computing the slope of the error signals
    verbose : boolean
        Prints information when true
    """

    if locks is None:
        locks = model.locks
    else:
        locks = [lock for lock in model.locks if lock.name in locks]
    
    for lock in locks:
        # Make sure readouts being used have their outputs enabled
        if type(lock.error_signal) is ReadoutDetectorOutput:
            lock.error_signal.readout.output_detectors = True

    # Use a flattened series analysis as it only creates one model
    # and xaxis resets all the parameters each time
    analysis = Series(
        *(
            Xaxis(lock.feedback, "lin", -d_dof, d_dof, 1, relative=True, name=lock.name)
            for lock in locks
        ),
        flatten=True,
    )
    sol = model.run(analysis)

    for lock in locks:
        lock_sol = sol[lock.name]
        x = lock_sol.x1
        error = lock_sol[lock.error_signal.name] + lock.offset
        grad = np.gradient(error, x[1] - x[0]).mean()
        if grad == 0:
            lock.gain = 0
        else:
            lock.gain = -1 / grad * gain_scale / 2

        if verbose:
            print(lock, lock.error_signal.name, lock.gain)

def find_operating_point(aligo):
    opt = aligo.run("""
        series(
            change(SRM.misaligned=True),
            maximize(Px, XARM.DC),
            change(SRM.misaligned=False),
        )
    """)
    aligo.run(
        finesse.analysis.actions.OptimiseRFReadoutPhaseDC(
            "SRCL",
            "rd_sr_f2",
            "CARM",
            "rd_sr_f1"
        )
    )
    set_lock_gains(aligo)

def lock_operation(aligo):
    find_operating_point(aligo)
    lock_sol = aligo.run(fac.RunLocks(max_iterations=500, exception_on_fail=False, display_progress=False))

def reset_model(pert=True):
    xarm_ligo = finesse.Model()
    xarm_ligo.parse(NOMINAL_KATSCRIPT)
    xarm_ligo.modes(maxtem=6, modes='even')
    xarm_ligo = perturb_model(xarm_ligo, 0.0001)
    
    finesse_g = xarm_ligo.optical_network
    
    for node in finesse_g.nodes():
        name = node.replace('.', '_')
        xarm_ligo.parse(f"pd p_{name} {node}")
        xarm_ligo.parse(f"fd f_{name} {node} f=0")
        xarm_ligo.parse(f"bp q_{name} {node} prop=q")
        
    return xarm_ligo

if __name__ == '__main__':
    num_dps = 30000
    count = 0
    
    xarm_ligo = finesse.Model()
    xarm_ligo.parse(NOMINAL_KATSCRIPT)
    xarm_ligo.modes(maxtem=6, modes='even')
    
    
    finesse_g = xarm_ligo.optical_network
    
    for node in finesse_g.nodes():
        name = node.replace('.', '_')
        xarm_ligo.parse(f"pd p_{name} {node}")
        xarm_ligo.parse(f"fd f_{name} {node} f=0")
        xarm_ligo.parse(f"bp q_{name} {node} prop=q")
    data = []
    
    while count < num_dps:
        if count % 100 == 0:
            xarm_ligo = reset_model()
        
        reset = False
        with warnings.catch_warnings():
            try:
                lock_operation(xarm_ligo)
                xarm_ligo.run()
            except:
                reset = True
        if reset or not xarm_ligo.cavSRX.is_stable:
            # If we reached a point where the cavity is unstable, restart the random walk
            xarm_ligo = reset_model()
            
            continue

        g = model_to_nx_port(xarm_ligo)
            
        data.append(g)
        count+=1
        xarm_ligo = perturb_model(xarm_ligo, 0.0001)
        
        print(count)
    with h5py.File('half_aligo_data.h5', 'w') as f:
        for i, graph in enumerate(data):
            # Serialize the graph object using pickle
            serialized_graph = pickle.dumps(graph)

            # Store the serialized graph in the HDF5 file
            f.create_dataset(f'sim_{i}', data=np.void(serialized_graph))