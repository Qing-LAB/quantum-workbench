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
# scipy.linalg.eigh: eigenvalue solver for Hermitian (symmetric) matrices.
#   Returns eigenvalues in ascending order and orthonormal eigenvectors.
#   We use eigh (not eig) because the Hamiltonian is Hermitian — this is a
#   physics constraint that guarantees real eigenvalues (= real energies)
#   and orthogonal eigenstates.
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Unit conversions — defined once, imported wherever needed.
# These let us convert between atomic units (natural for computation)
# and lab units (natural for experimentalists).
# ---------------------------------------------------------------------------
BOHR_PER_ANGSTROM = 1.8897259886
BOHR_PER_NM = 18.8973
HARTREE_TO_EV = 27.211386245988

# Consistent color palette for eigenstate plots — distinct, colorblind-friendly.
# Cycles through 8 colors and linestyles so multiple states are distinguishable.
STATE_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd',
                '#e377c2', '#ff7f0e', '#8c564b', '#17becf']
STATE_LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']


class QuantumSystem1D:
    """
    1D time-independent Schrödinger equation solver.

    Discretizes the Hamiltonian on a uniform grid using second-order
    finite differences and solves the resulting matrix eigenvalue problem.

    The key equation (atomic units, mass m):

        -1/(2m) d²φ/dx² + V(x) φ(x) = E φ(x)

    Discretized on N interior points, this becomes the N×N matrix problem:

        H · φ⃗ = E · φ⃗

    where H = T + V, with T the kinetic energy (tridiagonal) and V the
    potential energy (diagonal).

    Parameters
    ----------
    x_min, x_max : float
        Domain boundaries in bohr. The wavefunction is forced to zero at
        both boundaries (Dirichlet boundary conditions — appropriate for
        infinite walls or for soft potentials where the domain extends
        far enough that the wavefunction has decayed to negligible values).
    N : int
        Number of interior grid points. More points = finer grid = better
        accuracy, but larger matrix to diagonalize.
    V_func : callable
        V(x) → potential energy in hartree. Must accept and return arrays.
    mass : float, optional
        Particle mass in electron masses. Default: 1.0 (electron).
        Use 1836.15 for a proton, or a reduced mass for diatomic vibrations.
    """

    def __init__(self, x_min, x_max, N, V_func, mass=1.0):
        self.x_min = x_min
        self.x_max = x_max
        self.N = N
        self.mass = mass
        self.V_func = V_func

        # Grid: N equally spaced interior points. We exclude the boundaries
        # x_min and x_max because φ = 0 there (Dirichlet BCs).
        # Grid spacing: dx = (x_max - x_min) / (N + 1), where the +1 accounts
        # for the boundary points on each end.
        self.dx = (x_max - x_min) / (N + 1)
        # np.linspace: creates N evenly spaced points from x_min+dx to x_max-dx
        self.x = np.linspace(x_min + self.dx, x_max - self.dx, N)

        # Build the Hamiltonian matrix once at construction time
        self.H = self._build_hamiltonian()

        # Solutions (populated by solve())
        self.energies = None
        self.states = None

    def _build_hamiltonian(self):
        """
        Build H = T + V as an N×N matrix.

        Kinetic energy T: the second derivative -d²/dx² is approximated by
        the three-point stencil (φ_{j-1} - 2φ_j + φ_{j+1}) / dx², giving
        the tridiagonal matrix (1/2m·dx²) × tridiag(-1, 2, -1).

        Potential energy V: diagonal matrix with V(x_j) on the diagonal.
        """
        # --- Kinetic energy matrix (tridiagonal) ---
        # np.full(N, 2.0): array of N copies of 2.0 (main diagonal)
        diag_main = np.full(self.N, 2.0)
        # np.full(N-1, -1.0): array of N-1 copies of -1.0 (off-diagonals)
        diag_off = np.full(self.N - 1, -1.0)
        # np.diag(v, k): creates a matrix with v on the k-th diagonal.
        #   k=0: main diagonal, k=+1: superdiagonal, k=-1: subdiagonal.
        # Together these three terms build the tridiag(-1, 2, -1) matrix.
        T = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)
        # Scale by the kinetic energy prefactor: 1/(2m·dx²)
        # This comes from -1/(2m) × (-1, 2, -1)/dx² in the finite-difference
        # approximation of -1/(2m) d²/dx².
        T *= 1.0 / (2.0 * self.mass * self.dx**2)

        # --- Potential energy matrix (diagonal) ---
        # V(x_j) on the diagonal: each grid point sees the local potential.
        # np.diag converts a 1D array into a diagonal matrix.
        V = np.diag(self.V_func(self.x))

        return T + V

    def solve(self, n_states=10):
        """
        Solve H·φ = E·φ for the lowest n_states eigenstates.

        Uses scipy.linalg.eigh, which exploits the Hermitian symmetry of H
        to guarantee real eigenvalues and orthonormal eigenvectors.

        Returns
        -------
        energies : ndarray, shape (n_states,)
            Eigenvalues (energies) in hartree, sorted ascending.
        states : ndarray, shape (N, n_states)
            Normalized eigenvectors as columns. states[:, i] is the
            wavefunction for energy energies[i], sampled at the grid points.
        """
        # linalg.eigh returns ALL eigenvalues/vectors sorted by eigenvalue.
        # For large matrices, scipy.sparse.linalg.eigsh would be more efficient,
        # but eigh is simpler and sufficient for our grid sizes (N ≤ ~1000).
        eigenvalues, eigenvectors = linalg.eigh(self.H)

        # Keep only the lowest n_states (the physically interesting ones)
        self.energies = eigenvalues[:n_states]
        self.states = eigenvectors[:, :n_states]

        # Normalize on the grid: ∫|φ|² dx ≈ Σ|φ_j|² · dx = 1.
        # eigh returns unit-norm vectors in the discrete sense (Σ|φ_j|² = 1),
        # but we need the continuous normalization ∫|φ|² dx = 1, which
        # requires dividing by √(Σ|φ_j|² · dx).
        for i in range(n_states):
            norm = np.sqrt(np.sum(self.states[:, i]**2) * self.dx)
            self.states[:, i] /= norm

        return self.energies, self.states

    def x_full(self):
        """Grid including boundary points (where φ=0)."""
        # np.concatenate: joins arrays. We prepend x_min and append x_max
        # so that plots show the wavefunction going to zero at the walls.
        return np.concatenate([[self.x_min], self.x, [self.x_max]])

    def state_full(self, i):
        """Eigenstate i with boundary zeros appended."""
        return np.concatenate([[0], self.states[:, i], [0]])

    def plot_states(self, n_show=4, offset_by_energy=True, save_path=None):
        """
        Plot wavefunctions and probability densities side by side.

        Parameters
        ----------
        n_show : int
            Number of states to plot.
        offset_by_energy : bool
            If True, vertically offset each wavefunction by its energy.
            This produces the standard "energy level diagram" where each
            eigenstate sits at its energy level.
        save_path : str, optional
            If given, save the figure to this path.
        """
        if self.energies is None:
            self.solve(n_states=n_show)

        # Two side-by-side panels: wavefunctions (left) and |φ|² (right)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                        sharey=offset_by_energy)
        xf = self.x_full()

        for i in range(min(n_show, len(self.energies))):
            E = self.energies[i]
            phi = self.state_full(i)
            # Vertical offset: each state is drawn at its energy level
            offset = E if offset_by_energy else 0
            color = STATE_COLORS[i % len(STATE_COLORS)]
            ls = STATE_LINESTYLES[i % len(STATE_LINESTYLES)]

            # Left panel: wavefunction φ(x), offset vertically by energy
            ax1.plot(xf, phi + offset, color=color, linestyle=ls,
                     linewidth=1.5, label=f'n={i+1}, E={E:.4f} Ha')
            ax1.axhline(y=offset, color='gray', linewidth=0.3)

            # Right panel: probability density |φ(x)|², with shaded fill
            ax2.fill_between(xf, offset, phi**2 + offset,
                             color=color, alpha=0.15)
            ax2.plot(xf, phi**2 + offset, color=color, linestyle=ls,
                     linewidth=1.5, label=f'n={i+1}')
            ax2.axhline(y=offset, color='gray', linewidth=0.3)

        # Axis labels with units
        ax1.set_xlabel('x (bohr)')
        ax1.set_ylabel('$\\phi(x)$ + E (hartree)' if offset_by_energy
                        else '$\\phi(x)$')
        ax1.set_title('Wavefunctions')
        ax1.legend(fontsize=8)

        ax2.set_xlabel('x (bohr)')
        ax2.set_ylabel('$|\\phi(x)|^2$ + E' if offset_by_energy
                        else '$|\\phi(x)|^2$')
        ax2.set_title('Probability Densities')
        ax2.legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig

    def check_orthonormality(self, n_check=None):
        """
        Compute the overlap matrix ⟨φ_m|φ_n⟩.

        For correctly normalized, orthogonal eigenstates, this should be
        the identity matrix (1 on diagonal, 0 elsewhere). Deviations from
        δ_mn indicate numerical error from the finite grid.

        Returns
        -------
        overlap : ndarray, shape (n_check, n_check)
            Overlap matrix. overlap[m, n] = ∫ φ_m(x) φ_n(x) dx
            approximated as Σ φ_m(x_j) φ_n(x_j) dx.
        """
        if self.energies is None:
            raise RuntimeError("Call solve() first.")
        if n_check is None:
            n_check = len(self.energies)
        n_check = min(n_check, len(self.energies))

        overlap = np.zeros((n_check, n_check))
        for m in range(n_check):
            for n in range(n_check):
                # Grid integral: ⟨φ_m|φ_n⟩ = Σ_j φ_m(x_j) · φ_n(x_j) · dx
                overlap[m, n] = np.sum(
                    self.states[:, m] * self.states[:, n]
                ) * self.dx
        return overlap

    def count_nodes(self, i):
        """
        Count zero-crossings in eigenstate i.

        The n-th eigenstate of a 1D potential should have exactly n-1 nodes
        (the Sturm oscillation theorem). This provides a quick sanity check
        that our eigenstates are correctly ordered and physically reasonable.

        np.sign returns -1, 0, or +1. np.diff of the sign array is nonzero
        wherever the wavefunction changes sign (crosses zero).
        """
        phi = self.states[:, i]
        return int(np.sum(np.diff(np.sign(phi)) != 0))
