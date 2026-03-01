"""
quantum1d.py — 1D Quantum Eigenvalue Solver

Solve the time-independent Schrödinger equation in 1D using
finite-difference discretization and matrix diagonalization.

Units: atomic units (ħ = mₑ = e = 1)
    - Length: bohr (1 bohr = 0.5292 Å)
    - Energy: hartree (1 hartree = 27.21 eV)

Example
-------
    from quantum1d import QuantumSystem1D
    import numpy as np

    L = 18.90  # 1 nm in bohr
    system = QuantumSystem1D(0, L, 200, V_func=lambda x: np.zeros_like(x))
    energies, states = system.solve(n_states=5)
    system.plot_states()
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


# Unit conversions
BOHR_PER_ANGSTROM = 1.8897259886
BOHR_PER_NM = 18.8973
HARTREE_TO_EV = 27.211386245988


class QuantumSystem1D:
    """
    1D time-independent Schrödinger equation solver.

    Discretizes the Hamiltonian on a uniform grid using second-order
    finite differences and solves the resulting matrix eigenvalue problem.

    Parameters
    ----------
    x_min, x_max : float
        Domain boundaries in bohr.
    N : int
        Number of interior grid points.
    V_func : callable
        V(x) → potential energy in hartree. Must accept and return arrays.
    mass : float, optional
        Particle mass in electron masses. Default: 1.0 (electron).
    """

    def __init__(self, x_min, x_max, N, V_func, mass=1.0):
        self.x_min = x_min
        self.x_max = x_max
        self.N = N
        self.mass = mass
        self.V_func = V_func

        # Grid: interior points only (boundary values = 0)
        self.dx = (x_max - x_min) / (N + 1)
        self.x = np.linspace(x_min + self.dx, x_max - self.dx, N)

        # Build Hamiltonian
        self.H = self._build_hamiltonian()

        # Solutions (populated by solve())
        self.energies = None
        self.states = None

    def _build_hamiltonian(self):
        """Build H = T + V as an N×N matrix."""
        # Kinetic energy: (1/2m·dx²) × tridiag(-1, 2, -1)
        diag_main = np.full(self.N, 2.0)
        diag_off = np.full(self.N - 1, -1.0)
        T = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)
        T *= 1.0 / (2.0 * self.mass * self.dx**2)

        # Potential energy: diagonal
        V = np.diag(self.V_func(self.x))

        return T + V

    def solve(self, n_states=10):
        """
        Solve H·φ = E·φ for the lowest n_states eigenstates.

        Returns
        -------
        energies : ndarray, shape (n_states,)
        states : ndarray, shape (N, n_states)
            Normalized eigenvectors as columns.
        """
        eigenvalues, eigenvectors = linalg.eigh(self.H)

        self.energies = eigenvalues[:n_states]
        self.states = eigenvectors[:, :n_states]

        # Normalize on the grid
        for i in range(n_states):
            norm = np.sqrt(np.sum(self.states[:, i]**2) * self.dx)
            self.states[:, i] /= norm

        return self.energies, self.states

    def x_full(self):
        """Grid including boundary points (where φ=0)."""
        return np.concatenate([[self.x_min], self.x, [self.x_max]])

    def state_full(self, i):
        """Eigenstate i with boundary zeros appended."""
        return np.concatenate([[0], self.states[:, i], [0]])

    def plot_states(self, n_show=4, offset_by_energy=True, save_path=None):
        """
        Plot wavefunctions and probability densities.

        Parameters
        ----------
        n_show : int
            Number of states to plot.
        offset_by_energy : bool
            Offset each wavefunction vertically by its energy.
        save_path : str, optional
            If given, save the figure to this path.
        """
        if self.energies is None:
            self.solve(n_states=n_show)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                        sharey=offset_by_energy)
        xf = self.x_full()

        for i in range(min(n_show, len(self.energies))):
            E = self.energies[i]
            phi = self.state_full(i)
            offset = E if offset_by_energy else 0

            ax1.plot(xf, phi + offset, label=f'n={i+1}, E={E:.4f} Ha')
            ax1.axhline(y=offset, color='gray', linewidth=0.3)

            ax2.plot(xf, phi**2 + offset, label=f'n={i+1}')
            ax2.axhline(y=offset, color='gray', linewidth=0.3)

        for ax, title, ylabel in [
            (ax1, 'Wavefunctions', 'φ(x)'),
            (ax2, 'Probability Densities', '|φ(x)|²'),
        ]:
            ax.set_xlabel('x (bohr)')
            if offset_by_energy:
                ax.set_ylabel(f'{ylabel} + E (hartree)')
            else:
                ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def check_orthonormality(self, n_check=None):
        """Compute and print the overlap matrix <φ_m|φ_n>."""
        if self.energies is None:
            raise RuntimeError("Call solve() first.")
        if n_check is None:
            n_check = len(self.energies)
        n_check = min(n_check, len(self.energies))

        overlap = np.zeros((n_check, n_check))
        for m in range(n_check):
            for n in range(n_check):
                overlap[m, n] = np.sum(
                    self.states[:, m] * self.states[:, n]
                ) * self.dx
        return overlap

    def count_nodes(self, i):
        """Count zero-crossings in eigenstate i."""
        phi = self.states[:, i]
        return int(np.sum(np.diff(np.sign(phi)) != 0))
