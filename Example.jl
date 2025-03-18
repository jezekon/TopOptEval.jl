using TopOptEval


"""
Example usage of the FEMTetrahedral package showing
how to analyze a cantilever beam with custom boundary conditions.
"""
# function run_analysis()
    # 1. Import mesh from GMSH file
    grid = import_mesh("data/beam_linear_volume_mesh.vtu")
    grid = import_mesh("data/beam_linear_volume_mesh.msh")
    grid = import_mesh("data/beam_vfrac_04_smooth_1_Approx.vtu")

    # 2. Setup material model (steel)
    # λ, μ = create_material_model(210.0e9, 0.3)  # Young's modulus and Poisson's ratio
    λ, μ = create_material_model(1., 0.3)  # Young's modulus and Poisson's ratio

    # 3. Setup problem (initialize dof handler, cell values, matrices)
    dh, cellvalues, K, f = setup_problem(grid)

    # 4. Assemble stiffness matrix
    assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)

    # 5. Select nodes for boundary conditions
    # 5.1 Fixed nodes at x=0 (fixed in all directions)
    fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    
    # 5.2 Sliding boundary nodes at z=0, constrained in z direction only
    # sliding_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    # sliding_dofs = [3]  # Constraint only in z direction (DOF 3)
    
    # 5.3 Force application on a circular region at x=60mm
    force_nodes = select_nodes_by_circle(
        grid,
        [60.0, 0.0, 0.0],  # center point
        [1.0, 0.0, 0.0],   # normal direction (perpendicular to the face)
        5.0                # radius of 5mm
    )

    force_nodes2 = select_nodes_by_plane(grid, [60.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    # 6. Apply boundary conditions
    ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)
    # ch2 = apply_sliding_boundary!(K, f, dh, sliding_nodes, sliding_dofs)
    apply_force!(f, dh, collect(force_nodes), [1.0, 0.0, 0.0])  # 1N force in x direction
    
    # 7. Solve the system
    # u, energy = solve_system(K, f, ch1, ch2)
    u, energy, stress_field = solve_system(K, f, ch1)
    
    # 8. Print deformation energy
    println("Final deformation energy: $energy J")
    
    # 9. Export results to ParaView
    export_results(u, dh, "cantilever_beam_results_u")
    export_results(stress_field, dh, "cantilever_beam_results_stress")
    
    # return u, energy, dh
# end
1.342819351941161e-11 J
# Run the analysis
# u, energy, dh = run_analysis()
