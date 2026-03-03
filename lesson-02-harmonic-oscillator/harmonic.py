"""
harmonic.py — Analytical Harmonic Oscillator Reference

Exact wavefunctions, energies, classical probability distributions,
matrix elements, and perturbation theory formulas for the quantum
harmonic oscillator.

Units: atomic units (ħ = mₑ = e = 1)
    - Length: bohr
    - Energy: hartree
    - Frequency ω: hartree/ħ = hartree (since ħ = 1)

Example
-------
    from harmonic import HarmonicOscillator

    ho = HarmonicOscillator(omega=1.0)
    print(ho.energy(0))            # 0.5 hartree (zero-point energy)

    x = np.linspace(-6, 6, 500)
    phi_0 = ho.wavefunction(x, 0)  # Ground-state Gaussian
    phi_3 = ho.wavefunction(x, 3)  # 3rd excited state (3 nodes)
"""

import numpy as np
from scipy.special import eval_hermite, factorial
# eval_hermite(n, x): evaluates the physicist's Hermite polynomial H_n(x).
#   Numerically stable for n up to ~50. This is the workhorse that turns
#   our analytical formula φ_n(x) = N_n · H_n(αx) · exp(-α²x²/2) into numbers.
# factorial(n, exact=True): exact integer factorial n! (no floating-point loss).
#   We need this for the normalization constant 1/√(2^n · n!).


# ---------------------------------------------------------------------------
# Unit conversions — defined once, imported wherever needed.
# These let us convert between atomic units (natural for computation)
# and lab units (natural for experimentalists).
# ---------------------------------------------------------------------------
BOHR_PER_ANGSTROM = 1.8897259886
BOHR_PER_NM = 18.8973
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_WAVENUMBER = 219474.63  # 1 hartree = 219474.63 cm⁻¹ (for vibrational spectroscopy)


class HarmonicOscillator:
    """
    Analytical reference for the quantum harmonic oscillator.

    Potential:  V(x) = ½ω²x²    (atomic units, mass = 1)
    Energies:   E_n = ω(n + ½)
    States:     φ_n(x) = N_n · H_n(√ω · x) · exp(-ωx²/2)

    This class provides exact formulas — no numerical solving.
    We use it to validate our numerical solver (QuantumSystem1D from Lesson 01)
    and to supply analytical predictions for perturbation theory.

    Parameters
    ----------
    omega : float
        Angular frequency in atomic units (hartree/ħ).
        Sets both the energy scale (level spacing = ω) and
        the length scale (ground-state width ~ 1/√ω).
    """

    def __init__(self, omega=1.0):
        self.omega = omega

    # ------------------------------------------------------------------
    # Energies
    # ------------------------------------------------------------------

    def energy(self, n):
        """
        Exact energy of level n.

        Implements the boxed result from Part C:
            E_n = ω(n + ½)

        Parameters
        ----------
        n : int or array-like
            Quantum number(s), n = 0, 1, 2, ...
            Accepts arrays so we can compute all levels at once:
            ho.energy(np.arange(10)) gives E_0 through E_9.

        Returns
        -------
        float or ndarray
            Energy in hartree.
        """
        # np.asarray: lets this work for both a single int and a numpy array
        return self.omega * (np.asarray(n) + 0.5)

    # ------------------------------------------------------------------
    # Wavefunctions
    # ------------------------------------------------------------------

    def wavefunction(self, x, n):
        """
        Normalized eigenfunction φ_n(x) using Hermite polynomials.

        This is the explicit formula from Part D:
            φ_n(x) = N_n · H_n(αx) · exp(-α²x²/2)
        where α = √ω and N_n = (α / (√π · 2ⁿ · n!))^(1/2).

        The three factors have distinct physical roles:
        - H_n(αx): polynomial that creates the n nodes (oscillatory structure)
        - exp(-α²x²/2): Gaussian envelope (confinement by the parabolic well)
        - N_n: normalization so that ∫|φ_n|² dx = 1 (probability = 1)

        Parameters
        ----------
        x : ndarray
            Position values in bohr.
        n : int
            Quantum number (n ≥ 0).

        Returns
        -------
        ndarray
            φ_n(x), same shape as x.
        """
        # α = √ω sets the length scale: the ground-state Gaussian width is 1/α
        alpha = np.sqrt(self.omega)

        # ξ = αx is the dimensionless coordinate that appears in Hermite's equation
        xi = alpha * x

        # Normalization constant: N_n = (α / (√π · 2^n · n!))^{1/2}
        # factorial(n, exact=True) computes n! as an exact integer to avoid
        # floating-point overflow for large n (n! grows faster than 2^n)
        norm = (alpha / (np.sqrt(np.pi) * 2**n * factorial(n, exact=True)))**0.5

        # eval_hermite(n, xi): evaluates the physicist's Hermite polynomial H_n(ξ)
        # using a stable recurrence relation (not explicit polynomial coefficients,
        # which would overflow for large n)
        return norm * eval_hermite(n, xi) * np.exp(-xi**2 / 2)

    # ------------------------------------------------------------------
    # Classical probability distribution
    # ------------------------------------------------------------------

    def classical_probability(self, x, n):
        """
        Classical probability density P(x) for a harmonic oscillator
        with energy E_n = ω(n+½).

        From Part A.3: the classical particle spends time proportional to
        1/|velocity| at each position, giving:

            P(x) = 1 / (π √(x_turn² - x²))     for |x| < x_turn
                 = 0                               for |x| ≥ x_turn

        where x_turn = √(2E/ω²) is the classical turning point (where
        all energy is potential and the particle momentarily stops).

        We use this to compare against the quantum |φ_n(x)|² in Part E.4.

        Parameters
        ----------
        x : ndarray
            Position values in bohr.
        n : int
            Quantum number (sets the energy and hence turning points).

        Returns
        -------
        ndarray
            P(x), same shape as x. Zero outside turning points.
        """
        E = self.energy(n)
        # Classical turning point: V(x_turn) = E → ½ω²x_turn² = E
        x_turn = np.sqrt(2 * E / self.omega**2)

        # x_turn² - x²: positive inside the turning points, negative outside
        arg = x_turn**2 - x**2

        # Near the turning points (arg → 0), P(x) → ∞ (integrable singularity).
        # We cut off at a small eps to avoid division by zero.
        eps = 1e-12

        # np.maximum clamps negative values to eps BEFORE taking the sqrt.
        # This prevents RuntimeWarning from sqrt of negative numbers.
        # (np.where evaluates BOTH branches before choosing, so without this
        # clamp, sqrt would see negative arg values even though we'd discard them.)
        safe_arg = np.maximum(arg, eps)

        # np.where(condition, value_if_true, value_if_false):
        # Returns P(x) inside the turning points, 0 outside.
        P = np.where(arg > eps,
                     1.0 / (np.pi * np.sqrt(safe_arg)),
                     0.0)
        return P

    def turning_point(self, n):
        """
        Classical turning point x_turn for energy E_n.

        At the turning point, all energy is potential: V(x_turn) = E_n.
        Solving ½ω²x_turn² = ω(n+½) gives x_turn = √((2n+1)/ω).
        """
        E = self.energy(n)
        return np.sqrt(2 * E / self.omega**2)

    # ------------------------------------------------------------------
    # Matrix elements from ladder operator algebra (Part C / Part F)
    # ------------------------------------------------------------------

    def x_matrix_element(self, m, n):
        """
        Matrix element ⟨m|x̂|n⟩ from ladder operator algebra.

        From Part C: x̂ = (â⁺ + â⁻) / √(2ω), and the ladder operators
        only connect neighboring states (Δn = ±1). Therefore:

            ⟨m|x̂|n⟩ = √((n+1)/(2ω))  if m = n+1   (raising)
                      = √(n/(2ω))      if m = n-1   (lowering)
                      = 0               otherwise    (selection rule)

        This selection rule is why only Δn = ±1 transitions appear in
        infrared spectroscopy of harmonic oscillators.

        Parameters
        ----------
        m, n : int
            Quantum numbers.

        Returns
        -------
        float
        """
        if m == n + 1:
            return np.sqrt((n + 1) / (2 * self.omega))
        elif m == n - 1:
            return np.sqrt(n / (2 * self.omega))
        else:
            return 0.0

    def x4_expectation(self, n):
        """
        Expectation value ⟨n|x⁴|n⟩ computed via ladder operators.

        From Part F.3: express x⁴ in terms of (â⁺ + â⁻)⁴, expand,
        keep only diagonal (equal numbers of raising and lowering), and
        normal-order using [â⁻, â⁺] = 1. Result:

            ⟨n|x⁴|n⟩ = (3 / 4ω²)(2n² + 2n + 1)

        Used by perturbation_energy_first_order() to compute the
        first-order correction for the anharmonic potential V' = λx⁴.

        Parameters
        ----------
        n : int
            Quantum number.

        Returns
        -------
        float
        """
        return (3.0 / (4.0 * self.omega**2)) * (2 * n**2 + 2 * n + 1)

    def perturbation_energy_first_order(self, n, lam):
        """
        First-order perturbation theory correction for V' = λx⁴.

        From Part F.2–F.3:
            E_n^(1) = ⟨n|H'|n⟩ = λ⟨n|x⁴|n⟩ = (3λ / 4ω²)(2n² + 2n + 1)

        This is the leading correction when λ is small compared to ω.
        Valid when E_n^(1) ≪ ω (the level spacing). Breaks down for
        large λ or large n.

        Parameters
        ----------
        n : int
            Quantum number.
        lam : float
            Perturbation strength λ.

        Returns
        -------
        float
            Energy correction in hartree.
        """
        return lam * self.x4_expectation(n)

    # ------------------------------------------------------------------
    # Domain suggestion for numerical solvers
    # ------------------------------------------------------------------

    def suggest_domain(self, n_max, n_sigma=4.0):
        """
        Suggest a computational domain for the finite-difference solver.

        The HO wavefunction decays as exp(-ωx²/2) beyond the classical
        turning point — it never exactly reaches zero. Our solver imposes
        φ = 0 at the boundaries, so we need the domain to extend far enough
        that the truncation error is negligible.

        Strategy: start from the turning point of the highest state we want,
        then pad by n_sigma characteristic lengths (1/√ω). The wavefunction
        at distance d beyond the turning point is ~ exp(-ωd²/2), so
        n_sigma = 4 gives truncation ~ exp(-8) ≈ 3×10⁻⁴.

        Parameters
        ----------
        n_max : int
            Highest quantum number to be computed.
        n_sigma : float, optional
            Number of characteristic lengths beyond turning point. Default: 4.0.
            Larger values → more accurate but more grid points wasted on tails.

        Returns
        -------
        (x_min, x_max) : tuple of float
            Symmetric domain in bohr.
        """
        x_turn = self.turning_point(n_max)
        # Characteristic decay length of the Gaussian tail
        x_pad = n_sigma / np.sqrt(self.omega)
        x_max = x_turn + x_pad
        # Symmetric domain (the HO potential is symmetric about x=0)
        return (-x_max, x_max)

    def potential(self, x):
        """V(x) = ½ω²x² — the parabolic potential energy."""
        return 0.5 * self.omega**2 * x**2
