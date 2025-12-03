# Coupled Vibrational Spectroscopy During Tensile Deformation

A quantum mechanical simulation framework for modeling time-resolved infrared spectroscopy during mechanical deformation of polymer thin films.

## Overview

This project implements a full quantum mechanical treatment of vibrational mode coupling in mechanochromic materials under uniaxial tensile stress. The simulation combines:

- **Hamiltonian-based coupling dynamics**: Full diagonalization of the vibrational Hamiltonian including transition dipole, Fermi resonance, and mechanical coupling mechanisms
- **Time-resolved IR spectroscopy**: Generation of realistic IR spectra with Voigt lineshapes, stress-induced broadening, and dichroic effects
- **Mechanochromic analysis**: Tracking of eigenmode evolution, frequency splitting patterns, and intensity redistribution during deformation

## Key Features

- **Quantum mechanical rigor**: Constructs and diagonalizes strain-dependent Hamiltonian matrices to capture mode mixing and avoided crossings
- **Multiple coupling mechanisms**: Implements distance-dependent (r⁻³) transition dipole coupling, symmetry-mediated Fermi resonance, and mechanical coupling through shared bonds
- **Comprehensive visualization**: 12-panel analysis including spectral evolution maps, Hamiltonian matrices, eigenvector composition, and 3D waterfall plots
- **Modular architecture**: Object-oriented design with separate classes for material properties, coupling parameters, spectroscopy, and tensile testing

## Physical Model

The vibrational Hamiltonian in the frequency representation:
```
H = Σᵢ ωᵢ|i⟩⟨i| + Σᵢⱼ Vᵢⱼ(|i⟩⟨j| + |j⟩⟨i|)
```

where diagonal elements ωᵢ represent bare frequencies (with mechanochromic shifts) and off-diagonal elements Vᵢⱼ encode coupling strengths that evolve with strain.

## Applications

- Mechanochromic sensor design
- Polymer characterization under stress
- Vibrational coupling analysis in soft materials
- Education: demonstrates quantum mechanical principles in molecular spectroscopy

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

## Usage
```python
# Define material and vibrational modes
material = MaterialProperties(youngs_modulus=2e9, thickness=50e-6)
ir_bands = [IRBandProperties(center_freq=1720, intensity=1.0, ...)]
coupling_params = [CouplingParameters(mode1_idx=0, mode2_idx=1, coupling_strength=15.0, ...)]

# Create coupled system and run experiment
coupled_system = CoupledVibrationalSystem(ir_bands, coupling_params)
experiment = TandemTensileIRExperiment(material, coupled_system)
experiment.run_experiment(duration=10.0, num_spectra=50)

# Visualize results
experiment.plot_coupling_analysis()
```

## Results

The simulation successfully captures:
- Strain-dependent eigenfrequency evolution with characteristic non-parallel trajectories
- Coupling-induced frequency splittings that vary from initial to final states
- Mode mixing visualized through eigenvector composition matrices
- Realistic spectral evolution matching theoretical predictions

## Future Directions

- Non-linear mechanical response (yield, plasticity)
- Extension to 2D-IR spectroscopy
- Temperature-dependent coupling
- Validation against experimental mechanochromic IR data

## Author

Molecular Spectroscopy Final Project

## License

MIT License - feel free to use and modify for educational or research purposes.
