# Physics-Informed Neural Networks for Time-Series Reconstruction and Modelling

**Physics-Informed Neural Networks for Time-Series Reconstruction and Modelling with Sparse and Indirect Measurements**  
R. Rossi, M. Gelfusa, T. Craciunescu, N. Rutigliano, P. Gaudio, A. Murari,  
on behalf of JET contributors and EUROfusion Tokamak Exploitation Team  
*[journal name]* — DOI: [DOI]

---

## Overview

This repository contains the MATLAB source code for the synthetic case studies presented in the Supplementary Materials of the above publication. Each case study demonstrates the application of Physics-Informed Neural Networks (PINNs) to time-series reconstruction and system identification under conditions of data scarcity, measurement noise, partial observability, and incomplete physical knowledge.

All synthetic examples are based on the Lorenz system as a benchmark for nonlinear and chaotic dynamics. The PINN framework minimises a composite loss combining a data-consistency term and a physics residual term enforced at collocation points sampled via Sobol sequences. Training is performed using the Adam optimiser with learning rate decay, implemented in MATLAB Deep Learning Toolbox.

---

## Repository Structure

```
/
├── README.md
├── CaseA/
│   └── main_caseA.m
├── CaseB/
│   └── main_caseB.m
├── CaseC/
│   └── main_caseC.m
├── CaseD/
│   └── main_caseD.m
└── CaseE/
    └── main_caseE.m
```

Each folder is self-contained and includes the main script for the corresponding case study. Before running any example, navigate to the chrona-scope root folder and run `chronascope_init.m` to add all required paths to the MATLAB environment.

---

## Case Studies

### Case A — Time-series reconstruction from sparse direct measurements
Reconstruction of the full Lorenz state (x, y, z) from sparse and noisy direct measurements of all three variables. All model parameters are known. Serves as the reference benchmark.

### Case B — Time-series reconstruction with hidden variables
Reconstruction of the full Lorenz state from measurements of x(t) only. The variables y(t) and z(t) are completely unobserved. The physics constraints compensate for the absence of direct observations.

### Case C — Time-series reconstruction from indirect measurements
Reconstruction of the full Lorenz state from sparse measurements of two nonlinear indirect observables, f(t) = x + y and g(t) = x(1 + z)/50. No state variable is directly measurable.

### Case D — Parameter identification with incomplete physics and indirect measurements
Simultaneous reconstruction of the Lorenz state and identification of unknown model parameters (σ, ρ, β) from indirect measurements. The equations are augmented with spurious terms to test robustness against model misspecification.

### Case E — Attractor disentanglement from mixed dynamics
Simultaneous disentanglement and reconstruction of two independent dynamical systems — the Lorenz attractor and the Lotka–Volterra system — observed only through their superposition. Unknown parameters of both systems are identified during training using an adaptive physics weighting strategy.

---

## Requirements

- MATLAB R2021b or later
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox

---

## Usage

1. Clone or download the entire chrona-scope repository.
2. Open MATLAB, navigate to the chrona-scope root folder, and run `chronascope_init.m`.
3. Navigate to the desired case folder (e.g., `CaseA/`).
4. Run the main script (e.g., `main_caseA.m`).

Training progress and reconstructed trajectories are displayed in real time. Expected training time depends on the available hardware.

---

## Citation

If you use this code in your research, please cite:

> R. Rossi, M. Gelfusa, T. Craciunescu, N. Rutigliano, P. Gaudio, A. Murari et al.,
> *Physics-Informed Neural Networks for Time-Series Reconstruction and Modelling with Sparse and Indirect Measurements*,
> [journal name], DOI: [DOI]

---

## Contact

Corresponding author: [r.rossi@ing.uniroma2.it](mailto:r.rossi@ing.uniroma2.it)
