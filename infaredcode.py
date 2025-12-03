import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.linalg import eigh
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class MaterialProperties:
    """Material properties for thin film sample"""
    thickness: float = 50e-6  # m (50 microns)
    width: float = 5e-3  # m (5 mm)
    length: float = 20e-3  # m (20 mm)
    youngs_modulus: float = 2e9  # Pa (typical polymer)
    poisson_ratio: float = 0.35
    density: float = 1200  # kg/m³

@dataclass
class IRBandProperties:
    """Properties of an IR-active vibrational mode"""
    center_freq: float  # cm⁻¹
    intensity: float  # arbitrary units
    linewidth: float  # cm⁻¹ (FWHM)
    orientation_angle: float = 0.0  # degrees, initial orientation
    stress_shift_coeff: float = -5.0  # cm⁻¹/GPa (mechanochromic shift)
    stress_broadening_coeff: float = 2.0  # cm⁻¹/GPa
    transition_dipole_moment: float = 1.0  # Debye (for coupling calculations)
    local_strain_sensitivity: float = 1.0  # Sensitivity to local strain field

@dataclass
class CouplingParameters:
    """Defines coupling between two vibrational modes"""
    mode1_idx: int
    mode2_idx: int
    coupling_strength: float  # cm⁻¹
    coupling_type: str = 'transition_dipole'  # 'transition_dipole', 'fermi', 'mechanical'
    distance_dependent: bool = True  # If True, coupling changes with deformation
    stress_modulation_coeff: float = 0.5  # How stress affects coupling

class TensileTester:
    """Simulates uniaxial tensile deformation"""

    def __init__(self, material: MaterialProperties, strain_rate: float = 0.01):
        self.material = material
        self.strain_rate = strain_rate  # s⁻¹

    def apply_strain(self, time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate strain and stress as function of time"""
        strain = self.strain_rate * time
        # Linear elastic response (can be extended to non-linear)
        stress = self.material.youngs_modulus * strain
        return strain, stress

    def calculate_local_strain_field(self, global_strain: float, n_points: int = 100) -> np.ndarray:
        """
        Calculate local strain distribution (can include heterogeneity)
        For simple case, returns uniform field; can be extended for gradients
        """
        # Add small spatial heterogeneity (realistic for thin films)
        local_strains = global_strain * (1 + 0.1 * np.random.randn(n_points))
        return local_strains

class CoupledVibrationalSystem:
    """
    Handles vibrational coupling using full quantum mechanical Hamiltonian formalism.

    The vibrational Hamiltonian in the harmonic oscillator basis is:
    Ĥ = Σᵢ ℏωᵢ(â†ᵢâᵢ + 1/2) + Σᵢⱼ Vᵢⱼ(â†ᵢâⱼ + âᵢâ†ⱼ)

    For spectroscopy, we work in the frequency representation (cm⁻¹):
    H = Σᵢ ωᵢ|i⟩⟨i| + Σᵢⱼ Vᵢⱼ(|i⟩⟨j| + |j⟩⟨i|)
    """

    def __init__(self, bands: List[IRBandProperties],
                 coupling_params: List[CouplingParameters]):
        self.bands = bands
        self.coupling_params = coupling_params
        self.n_modes = len(bands)
        self.hbar = 5.3088e-12  # ℏ in cm⁻¹·s (for conversion if needed)

    def build_hamiltonian(self, stress: float, strain: float) -> np.ndarray:
        """
        Construct vibrational Hamiltonian matrix including coupling terms.
        H_ij = ω_i δ_ij + V_ij(1-δ_ij)
        where ω_i are diagonal energies and V_ij are coupling matrix elements
        """
        H = np.zeros((self.n_modes, self.n_modes))
        stress_gpa = stress / 1e9

        # Diagonal elements: bare frequencies with stress shifts
        for i, band in enumerate(self.bands):
            freq_shifted = band.center_freq + band.stress_shift_coeff * stress_gpa
            H[i, i] = freq_shifted

        # Off-diagonal elements: coupling terms
        for coupling in self.coupling_params:
            i, j = coupling.mode1_idx, coupling.mode2_idx

            # Base coupling strength
            V_ij = coupling.coupling_strength

            # Stress modulation of coupling
            if coupling.coupling_type == 'transition_dipole':
                # TDC scales with distance (r^-3 for dipole-dipole)
                # Strain changes distance: r(ε) = r0(1 + ε)
                distance_factor = (1 + strain)**(-3) if coupling.distance_dependent else 1.0
                V_ij *= distance_factor

            elif coupling.coupling_type == 'fermi':
                # Fermi resonance can be enhanced or suppressed by stress
                # due to symmetry changes
                V_ij *= (1 + coupling.stress_modulation_coeff * stress_gpa)

            elif coupling.coupling_type == 'mechanical':
                # Mechanical coupling through shared bonds/atoms
                # Typically increases with stress
                V_ij *= (1 + coupling.stress_modulation_coeff * stress_gpa)

            H[i, j] = V_ij
            H[j, i] = V_ij  # Hermitian matrix

        return H

    def diagonalize_hamiltonian(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize Hamiltonian to get eigenmodes (normal modes) and eigenfrequencies
        Returns: (eigenfrequencies, eigenvectors)
        """
        eigenvalues, eigenvectors = eigh(H)
        return eigenvalues, eigenvectors

    def calculate_intensities(self, eigenvectors: np.ndarray, strain: float) -> np.ndarray:
        """
        Calculate intensities of eigenmodes based on transition dipole moments
        and orientation effects
        """
        intensities = np.zeros(self.n_modes)

        for i in range(self.n_modes):
            # Intensity is |Σ_j c_ij μ_j|² where c_ij are eigenvector components
            # and μ_j are transition dipole moments
            intensity = 0
            for j, band in enumerate(self.bands):
                # Orientation factor
                orientation_factor = self.calculate_orientation_factor(band, strain)
                # Weighted contribution
                intensity += (eigenvectors[j, i]**2 * band.intensity *
                            band.transition_dipole_moment * orientation_factor)
            intensities[i] = intensity

        return intensities

    def calculate_orientation_factor(self, band: IRBandProperties, strain: float) -> float:
        """
        Calculate dichroic ratio change due to molecular orientation.
        Assumes affine deformation and Herman's orientation function.
        """
        orientation_param = np.tanh(2 * strain * band.local_strain_sensitivity)
        dichroic_ratio = 1/3 + (2/3) * orientation_param
        return dichroic_ratio

    def calculate_linewidths(self, stress: float) -> np.ndarray:
        """Calculate linewidths with stress broadening"""
        stress_gpa = stress / 1e9
        linewidths = np.array([
            band.linewidth + band.stress_broadening_coeff * stress_gpa
            for band in self.bands
        ])
        return linewidths

class TransientIRSpectroscopy:
    """Models IR spectroscopy with vibrational coupling"""

    def __init__(self, wavenumber_range: Tuple[float, float], resolution: float = 1.0):
        self.wn_min, self.wn_max = wavenumber_range
        self.resolution = resolution
        self.wavenumbers = np.arange(self.wn_min, self.wn_max, self.resolution)

    def lorentzian(self, x: np.ndarray, center: float, width: float, intensity: float) -> np.ndarray:
        """Lorentzian lineshape for IR absorption band"""
        gamma = width / 2
        return intensity * (gamma**2) / ((x - center)**2 + gamma**2)

    def voigt_approx(self, x: np.ndarray, center: float, gamma_l: float,
                     gamma_g: float, intensity: float) -> np.ndarray:
        """
        Approximate Voigt profile (convolution of Lorentzian and Gaussian)
        Useful for more realistic lineshapes under stress
        """
        # Pseudo-Voigt approximation
        eta = gamma_l / (gamma_l + gamma_g)  # Mixing parameter
        lorentz = self.lorentzian(x, center, gamma_l, 1.0)
        gauss = intensity * np.exp(-((x - center) / gamma_g)**2)
        return intensity * (eta * lorentz + (1 - eta) * gauss)

    def generate_coupled_spectrum(self, coupled_system: CoupledVibrationalSystem,
                                  stress: float, strain: float,
                                  add_noise: bool = True) -> Tuple[np.ndarray, dict]:
        """Generate IR spectrum including coupling effects"""

        # Build and diagonalize Hamiltonian
        H = coupled_system.build_hamiltonian(stress, strain)
        eigenfreqs, eigenvectors = coupled_system.diagonalize_hamiltonian(H)

        # Calculate intensities and linewidths of eigenmodes
        intensities = coupled_system.calculate_intensities(eigenvectors, strain)
        linewidths = coupled_system.calculate_linewidths(stress)

        # Generate spectrum from eigenmodes
        spectrum = np.zeros_like(self.wavenumbers)

        for i in range(len(eigenfreqs)):
            # Add stress-dependent inhomogeneous broadening (Gaussian component)
            gamma_g = 0.3 * linewidths[i] * (1 + 0.5 * strain)

            spectrum += self.voigt_approx(self.wavenumbers, eigenfreqs[i],
                                         linewidths[i], gamma_g, intensities[i])

        # Add baseline and noise
        baseline = 0.05 * np.ones_like(spectrum)
        if add_noise:
            noise = 0.01 * np.random.randn(len(spectrum))
            spectrum += baseline + noise
        else:
            spectrum += baseline

        # Store diagnostic information
        diagnostics = {
            'hamiltonian': H,
            'eigenfrequencies': eigenfreqs,
            'eigenvectors': eigenvectors,
            'intensities': intensities,
            'linewidths': linewidths,
            'bare_frequencies': [band.center_freq for band in coupled_system.bands]
        }

        return spectrum, diagnostics

class TandemTensileIRExperiment:
    """Main experimental setup with coupled vibrational analysis"""

    def __init__(self, material: MaterialProperties,
                 coupled_system: CoupledVibrationalSystem,
                 wavenumber_range: Tuple[float, float] = (1600, 1800)):
        self.material = material
        self.coupled_system = coupled_system
        self.tensile = TensileTester(material)
        self.ir_spec = TransientIRSpectroscopy(wavenumber_range)

        # Storage for time-resolved data
        self.time_points = None
        self.strain_data = None
        self.stress_data = None
        self.spectra_stack = None
        self.diagnostics_stack = []

    def run_experiment(self, duration: float = 10.0, num_spectra: int = 50):
        """Run time-resolved tensile IR experiment with coupling analysis"""
        self.time_points = np.linspace(0, duration, num_spectra)
        self.strain_data, self.stress_data = self.tensile.apply_strain(self.time_points)

        # Collect spectra at each time point
        self.spectra_stack = np.zeros((num_spectra, len(self.ir_spec.wavenumbers)))
        self.diagnostics_stack = []

        for i, (strain, stress) in enumerate(zip(self.strain_data, self.stress_data)):
            spectrum, diagnostics = self.ir_spec.generate_coupled_spectrum(
                self.coupled_system, stress, strain
            )
            self.spectra_stack[i] = spectrum
            self.diagnostics_stack.append(diagnostics)

        print(f"Experiment complete: {num_spectra} spectra collected")
        print(f"Strain range: {self.strain_data[0]:.4f} to {self.strain_data[-1]:.4f}")
        print(f"Stress range: {self.stress_data[0]/1e6:.2f} to {self.stress_data[-1]/1e6:.2f} MPa")
        print(f"\n=== Coupling Analysis ===")
        self._print_coupling_effects()

    def _print_coupling_effects(self):
        """Analyze and print coupling effects"""
        initial_diag = self.diagnostics_stack[0]
        final_diag = self.diagnostics_stack[-1]

        print(f"Number of coupled modes: {len(self.coupled_system.bands)}")
        print(f"Number of coupling interactions: {len(self.coupled_system.coupling_params)}")

        print("\nInitial state (ε=0):")
        print("  Bare frequencies:", [f"{f:.2f}" for f in initial_diag['bare_frequencies']])
        print("  Eigenfrequencies:", [f"{f:.2f}" for f in initial_diag['eigenfrequencies']])

        print(f"\nFinal state (ε={self.strain_data[-1]:.4f}):")
        print("  Eigenfrequencies:", [f"{f:.2f}" for f in final_diag['eigenfrequencies']])

        # Calculate splitting patterns
        for i in range(len(initial_diag['eigenfrequencies']) - 1):
            initial_split = initial_diag['eigenfrequencies'][i+1] - initial_diag['eigenfrequencies'][i]
            final_split = final_diag['eigenfrequencies'][i+1] - final_diag['eigenfrequencies'][i]
            print(f"\nSplitting between modes {i} and {i+1}:")
            print(f"  Initial: {initial_split:.2f} cm⁻¹")
            print(f"  Final: {final_split:.2f} cm⁻¹")
            print(f"  Change: {final_split - initial_split:.2f} cm⁻¹")

    def extract_eigenmode_evolution(self) -> dict:
        """Extract evolution of eigenmodes throughout deformation"""
        n_modes = len(self.coupled_system.bands)
        results = {
            'time': self.time_points,
            'strain': self.strain_data,
            'stress': self.stress_data,
            'eigenfrequencies': np.zeros((len(self.time_points), n_modes)),
            'intensities': np.zeros((len(self.time_points), n_modes)),
            'splittings': []
        }

        for i, diag in enumerate(self.diagnostics_stack):
            results['eigenfrequencies'][i] = diag['eigenfrequencies']
            results['intensities'][i] = diag['intensities']

        # Calculate frequency splittings between adjacent modes
        for i in range(n_modes - 1):
            splitting = (results['eigenfrequencies'][:, i+1] -
                        results['eigenfrequencies'][:, i])
            results['splittings'].append(splitting)

        return results

    def plot_coupling_analysis(self):
        """Comprehensive visualization including coupling effects"""
        fig = plt.figure(figsize=(18, 12))

        evolution = self.extract_eigenmode_evolution()
        n_modes = len(self.coupled_system.bands)

        # 1. Stress-strain curve
        ax1 = plt.subplot(3, 4, 1)
        ax1.plot(self.strain_data * 100, self.stress_data / 1e6, 'b-', linewidth=2)
        ax1.set_xlabel('Strain (%)', fontsize=10)
        ax1.set_ylabel('Stress (MPa)', fontsize=10)
        ax1.set_title('Mechanical Response', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Eigenfrequency evolution
        ax2 = plt.subplot(3, 4, 2)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_modes))
        for i in range(n_modes):
            ax2.plot(self.strain_data * 100, evolution['eigenfrequencies'][:, i],
                    '-o', color=colors[i], label=f'Mode {i+1}', linewidth=2, markersize=3)
        ax2.set_xlabel('Strain (%)', fontsize=10)
        ax2.set_ylabel('Eigenfrequency (cm⁻¹)', fontsize=10)
        ax2.set_title('Eigenmode Frequencies vs Strain', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Frequency splittings evolution
        ax3 = plt.subplot(3, 4, 3)
        for i, splitting in enumerate(evolution['splittings']):
            ax3.plot(self.strain_data * 100, splitting, '-o',
                    label=f'Modes {i+1}-{i+2}', linewidth=2, markersize=3)
        ax3.set_xlabel('Strain (%)', fontsize=10)
        ax3.set_ylabel('Frequency Splitting (cm⁻¹)', fontsize=10)
        ax3.set_title('Coupling-Induced Splitting', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Intensity evolution
        ax4 = plt.subplot(3, 4, 4)
        for i in range(n_modes):
            ax4.plot(self.strain_data * 100, evolution['intensities'][:, i],
                    '-o', color=colors[i], label=f'Mode {i+1}', linewidth=2, markersize=3)
        ax4.set_xlabel('Strain (%)', fontsize=10)
        ax4.set_ylabel('Intensity', fontsize=10)
        ax4.set_title('Eigenmode Intensities', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # 5. 2D spectral evolution map
        ax5 = plt.subplot(3, 4, 5)
        extent = [self.ir_spec.wavenumbers[0], self.ir_spec.wavenumbers[-1],
                  self.strain_data[0] * 100, self.strain_data[-1] * 100]
        im = ax5.imshow(self.spectra_stack, aspect='auto', origin='lower',
                       extent=extent, cmap='viridis', interpolation='bilinear')
        # Overlay eigenfrequency tracks
        for i in range(n_modes):
            ax5.plot(evolution['eigenfrequencies'][:, i], self.strain_data * 100,
                    'r--', linewidth=1.5, alpha=0.7)
        ax5.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
        ax5.set_ylabel('Strain (%)', fontsize=10)
        ax5.set_title('Spectral Evolution with Mode Tracking', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax5, label='Absorbance')

        # 6. Initial vs final spectra
        ax6 = plt.subplot(3, 4, 6)
        ax6.plot(self.ir_spec.wavenumbers, self.spectra_stack[0], 'b-',
                label=f'Initial (ε={self.strain_data[0]:.3f})', linewidth=2)
        ax6.plot(self.ir_spec.wavenumbers, self.spectra_stack[-1], 'r-',
                label=f'Final (ε={self.strain_data[-1]:.3f})', linewidth=2)
        # Mark eigenfrequencies
        for i in range(n_modes):
            ax6.axvline(self.diagnostics_stack[0]['eigenfrequencies'][i],
                       color='b', linestyle=':', alpha=0.5)
            ax6.axvline(self.diagnostics_stack[-1]['eigenfrequencies'][i],
                       color='r', linestyle=':', alpha=0.5)
        ax6.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
        ax6.set_ylabel('Absorbance', fontsize=10)
        ax6.set_title('Spectral Comparison', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        # 7. Hamiltonian matrix evolution (initial)
        ax7 = plt.subplot(3, 4, 7)
        H_initial = self.diagnostics_stack[0]['hamiltonian']
        im7 = ax7.imshow(H_initial, cmap='RdBu', aspect='auto')
        ax7.set_xlabel('Mode', fontsize=10)
        ax7.set_ylabel('Mode', fontsize=10)
        ax7.set_title('Initial Hamiltonian Matrix', fontsize=11, fontweight='bold')
        plt.colorbar(im7, ax=ax7, label='cm⁻¹')

        # 8. Hamiltonian matrix evolution (final)
        ax8 = plt.subplot(3, 4, 8)
        H_final = self.diagnostics_stack[-1]['hamiltonian']
        im8 = ax8.imshow(H_final, cmap='RdBu', aspect='auto')
        ax8.set_xlabel('Mode', fontsize=10)
        ax8.set_ylabel('Mode', fontsize=10)
        ax8.set_title('Final Hamiltonian Matrix', fontsize=11, fontweight='bold')
        plt.colorbar(im8, ax=ax8, label='cm⁻¹')

        # 9. Eigenvector composition (mixing) - initial
        ax9 = plt.subplot(3, 4, 9)
        eigvec_initial = self.diagnostics_stack[0]['eigenvectors']
        im9 = ax9.imshow(np.abs(eigvec_initial)**2, cmap='hot', aspect='auto', vmin=0, vmax=1)
        ax9.set_xlabel('Eigenmode', fontsize=10)
        ax9.set_ylabel('Bare Mode', fontsize=10)
        ax9.set_title('Initial Mode Mixing (|c_ij|²)', fontsize=11, fontweight='bold')
        plt.colorbar(im9, ax=ax9, label='Weight')

        # 10. Eigenvector composition (mixing) - final
        ax10 = plt.subplot(3, 4, 10)
        eigvec_final = self.diagnostics_stack[-1]['eigenvectors']
        im10 = ax10.imshow(np.abs(eigvec_final)**2, cmap='hot', aspect='auto', vmin=0, vmax=1)
        ax10.set_xlabel('Eigenmode', fontsize=10)
        ax10.set_ylabel('Bare Mode', fontsize=10)
        ax10.set_title('Final Mode Mixing (|c_ij|²)', fontsize=11, fontweight='bold')
        plt.colorbar(im10, ax=ax10, label='Weight')

        # 11. Coupling strength evolution
        ax11 = plt.subplot(3, 4, 11)
        for coupling in self.coupled_system.coupling_params:
            i, j = coupling.mode1_idx, coupling.mode2_idx
            coupling_evolution = []
            for k, strain in enumerate(self.strain_data):
                H = self.diagnostics_stack[k]['hamiltonian']
                coupling_evolution.append(H[i, j])
            ax11.plot(self.strain_data * 100, coupling_evolution, '-o',
                     label=f'V_{i+1},{j+1}', linewidth=2, markersize=3)
        ax11.set_xlabel('Strain (%)', fontsize=10)
        ax11.set_ylabel('Coupling Strength (cm⁻¹)', fontsize=10)
        ax11.set_title('Strain-Modulated Coupling', fontsize=11, fontweight='bold')
        ax11.legend(fontsize=8)
        ax11.grid(True, alpha=0.3)

        # 12. 3D waterfall plot
        ax12 = plt.subplot(3, 4, 12, projection='3d')
        X, Y = np.meshgrid(self.ir_spec.wavenumbers, self.strain_data * 100)
        surf = ax12.plot_surface(X, Y, self.spectra_stack, cmap='plasma',
                                alpha=0.8, linewidth=0, antialiased=True)
        ax12.set_xlabel('Wavenumber (cm⁻¹)', fontsize=8)
        ax12.set_ylabel('Strain (%)', fontsize=8)
        ax12.set_zlabel('Absorbance', fontsize=8)
        ax12.set_title('3D Spectral Evolution', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.show()


# Example usage with coupled modes
if __name__ == "__main__":
    # Define material
    material = MaterialProperties(
        thickness=50e-6,
        youngs_modulus=2e9,
        poisson_ratio=0.35
    )

    # Define IR-active bands that will couple
    ir_bands = [
        IRBandProperties(
            center_freq=1720,  # C=O stretch (amide I)
            intensity=1.0,
            linewidth=12,
            stress_shift_coeff=-8.0,
            stress_broadening_coeff=2.5,
            transition_dipole_moment=1.2,
            local_strain_sensitivity=1.0
        ),
        IRBandProperties(
            center_freq=1655,  # C=O stretch (nearby group)
            intensity=0.8,
            linewidth=15,
            stress_shift_coeff=-6.0,
            stress_broadening_coeff=3.0,
            transition_dipole_moment=1.0,
            local_strain_sensitivity=0.9
        ),
        IRBandProperties(
            center_freq=1690,  # Intermediate mode
            intensity=0.5,
            linewidth=10,
            stress_shift_coeff=-7.0,
            stress_broadening_coeff=2.0,
            transition_dipole_moment=0.8,
            local_strain_sensitivity=1.1
        )
    ]

    # Define coupling between modes
    coupling_params = [
        CouplingParameters(
            mode1_idx=0,
            mode2_idx=1,
            coupling_strength=15.0,  # Strong coupling
            coupling_type='transition_dipole',
            distance_dependent=True,
            stress_modulation_coeff=0.3
        ),
        CouplingParameters(
            mode1_idx=1,
            mode2_idx=2,
            coupling_strength=8.0,  # Moderate coupling
            coupling_type='fermi',
            distance_dependent=False,
            stress_modulation_coeff=0.5
        ),
        CouplingParameters(
            mode1_idx=0,
            mode2_idx=2,
            coupling_strength=5.0,  # Weak coupling
            coupling_type='mechanical',
            distance_dependent=True,
            stress_modulation_coeff=0.8
        )
    ]

    # Create coupled system
    coupled_system = CoupledVibrationalSystem(ir_bands, coupling_params)

    # Create and run experiment
    experiment = TandemTensileIRExperiment(
        material=material,
        coupled_system=coupled_system,
        wavenumber_range=(1600, 1800)
    )

    experiment.run_experiment(duration=10.0, num_spectra=50)

    # Comprehensive visualization with coupling analysis
    experiment.plot_coupling_analysis()

    # Extract detailed eigenmode evolution
    evolution = experiment.extract_eigenmode_evolution()
    print("\n=== Eigenmode Statistics ===")
    for i in range(len(ir_bands)):
        initial_freq = evolution['eigenfrequencies'][0, i]
        final_freq = evolution['eigenfrequencies'][-1, i]
        print(f"\nMode {i+1}:")
        print(f"  Initial eigenfrequency: {initial_freq:.2f} cm⁻¹")
        print(f"  Final eigenfrequency: {final_freq:.2f} cm⁻¹")
        print(f"  Total shift: {final_freq - initial_freq:.2f} cm⁻¹")
