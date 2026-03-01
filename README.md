# Quantum Workbench

A hands-on journey through quantum mechanics — pairing conceptual understanding
with computational capability from the ground up.

## Philosophy

- **Concept → Math → Code → Validation**: every lesson follows this arc
- **The Translation Layer**: we treat the bridge between mathematical formulation
  and numerical computation as a first-class skill — units, discretization,
  dimensional analysis, and sanity-checking are not afterthoughts
- **Good Design Over Complexity**: clean, readable code that maps directly to
  the physics. No black boxes.
- **Build Up Gradually**: start with systems we can solve analytically, validate
  our tools, then tackle problems that *require* numerical methods

## Target Audience

Scientists and engineers with some programming experience (Python, C/C++) who
want to develop quantitative quantum mechanics skills — not just textbook
understanding, but the ability to translate real-world problems into simplified
models and produce physically meaningful numerical results.

## Lessons

| Lesson | Topic | Key Skills |
|--------|-------|------------|
| 01 | [Particle in a 1D Box](lesson-01-particle-in-a-box/) | Schrödinger equation, discretization, eigenvalue problems, units |

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib

```bash
pip install numpy scipy matplotlib
```

Future lessons will introduce [PySCF](https://pyscf.org/) for molecular quantum
chemistry calculations.

## License

MIT
