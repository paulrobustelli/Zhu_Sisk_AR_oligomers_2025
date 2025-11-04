# Zhu_Sisk_AR_oligomers_2025
Code Accompanying "Molecular mechanisms of small molecules that stabilize oligomers of a phase-separating intrinsically disordered protein"

This repository contains scripts and Jupyter notebooks for analyzing the **Tau-5<sub>R2</sub>** dimer simulations in the **apo**, **EPI-002–bound**, and **1aa–bound** conditions.

The molecular dynamics trajectories used in this manuscript are stored separately due to file size limits and will be distributed via a shared download link (Dropbox / Globus / Zenodo).  
Place all downloaded trajectory files in a folder named `trajectory/` before running any notebooks.

## 📖 Reading the Trajectories in Python

### Load protein-only trajectories

```python
import mdtraj as md

traj_apo = md.load('trajectory/apo.protein.100us.stride100.dcd',
                   top='trajectory/R2_dimer.pdb')

traj_E2 = md.load('trajectory/E2.protein.100us.stride100.dcd',
                  top='trajectory/R2_dimer.pdb')

traj_1AA = md.load('trajectory/1AA.protein.85us.stride100.dcd',
                   top='trajectory/R2_dimer.pdb')
```
### Load protein+ligand trajectories
```python
import mdtraj as md

traj_E2_lig = md.load('trajectory/E2.protein.ligand.100us.stride100.dcd',
                      top='trajectory/R2_dimer_epi002.pdb')

traj_1AA_lig = md.load('trajectory/1AA.protein.ligand.85us.stride100.dcd',
                       top='trajectory/R2_dimer_1aa.pdb')
```
