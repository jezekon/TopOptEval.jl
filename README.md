# TopOptEval
Small Julia package for evaluating topology-optimized structures using strain energy analysis.
## Features
- Analysis of optimized designs via strain energy metrics
- Comparison of strain energy between raw SIMP results and post-processed geometries
- Import: GMSH (.msh), VTK (.vtu) with density data
- FEA: Linear elastic analysis with variable material properties
- Boundary conditions: Fixed, sliding, forces on geometric selections
- Export: VTK (.vtu) for displacement, stress fields, and boundary visualization
- Volume calculation: Standard and density-weighted

Built on Ferrite.jl with support for tetrahedral and hexahedral elements.
For usage examples, see tests.
