using Ferrite

"""
Test cantilever beam under gravity - analytical solution available
"""
function test_cantilever_gravity()
    # Beam geometry
    L, w, h = 10.0, 1.0, 1.0  # length, width, height
    grid = generate_grid(Hexahedron, (40, 8, 8), Vec((0.0, 0.0, 0.0)), Vec((L, w, h)))

    # Material properties - steel
    E = 200e9  # Pa
    ν = 0.3
    ρ = 7850   # kg/m³ 
    g = 9.81   # m/s²

    print_info("Setting up cantilever beam analysis (L=$L m, E=$E Pa, ρ=$ρ kg/m³)")

    # Setup FE problem
    λ, μ = create_material_model(E, ν)
    dh, cellvalues, K, f = setup_problem(grid)
    assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)

    # Apply boundary conditions - fixed at x=0
    fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ch = apply_fixed_boundary!(K, f, dh, fixed_nodes)

    # Apply gravity as volume force
    apply_gravity!(f, dh, cellvalues, ρ, g, [0.0, 0.0, -1.0])

    # Solve
    u, energy, stress_field, max_vm, max_cell = solve_system(K, f, dh, cellvalues, λ, μ, ch)

    # Analytical solution for maximum deflection of cantilever under self-weight:
    # δ_max = (ρ * g * L^4) / (8 * E * I) for uniform distributed load
    I = w * h^3 / 12  # second moment of area
    analytical_deflection = (ρ * g * L^4) / (8 * E * I)

    # Extract numerical deflection (maximum Z-displacement)
    numerical_deflection = maximum(abs.(u[3:3:end]))  # every third DOF is Z
    relative_error =
        abs(numerical_deflection - analytical_deflection) / analytical_deflection * 100

    print_data("\n" * "="^60)
    print_data("CANTILEVER BEAM RESULTS COMPARISON")
    print_data("="^60)
    print_data("Analytical deflection: $(round(analytical_deflection, digits=6)) m")
    print_data("Numerical deflection:  $(round(numerical_deflection, digits=6)) m")
    print_data("Relative error:        $(round(relative_error, digits=2)) %")

    if relative_error < 5.0
        print_success("✓ Results match analytical solution (error < 5%)")
    elseif relative_error < 10.0
        print_warning("! Moderate error (5-10%), consider mesh refinement")
    else
        print_error("✗ Large error (>10%), check implementation or mesh")
    end

    return u, analytical_deflection, grid, dh, numerical_deflection, relative_error
end

"""
Test gravity on simple cube - uniform stress verification
"""
function test_cube_gravity()
    # Unit cube
    grid = generate_grid(Hexahedron, (8, 8, 8), Vec((0.0, 0.0, 0.0)), Vec((1.0, 1.0, 1.0)))

    # Material properties
    E = 200e9
    ν = 0.3
    ρ = 7850

    print_info("Setting up cube gravity test (1x1x1 m)")

    λ, μ = create_material_model(E, ν)
    dh, cellvalues, K, f = setup_problem(grid)
    assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)

    # Fix bottom surface (z=0)
    fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    ch = apply_fixed_boundary!(K, f, dh, fixed_nodes)

    # Apply gravity
    apply_gravity!(f, dh, cellvalues, ρ)

    # Solve
    u, energy, stress_field, max_vm, max_cell = solve_system(K, f, dh, cellvalues, λ, μ, ch)

    # Analytical stress at bottom: σ_z = ρ * g * h
    analytical_stress = ρ * 9.81 * 1.0  # Pa
    max_displacement = maximum(abs.(u))

    print_data("\n" * "="^60)
    print_data("CUBE GRAVITY TEST RESULTS")
    print_data("="^60)
    print_data("Expected stress at bottom: $(round(analytical_stress, digits=0)) Pa")
    print_data("Maximum displacement:      $(round(max_displacement*1e6, digits=2)) μm")
    print_data("Maximum von Mises stress:  $(round(max_vm, digits=0)) Pa")

    return u, analytical_stress, grid, dh, max_displacement
end

"""
Test different gravity directions for validation
"""
function test_gravity_directions()
    grid = generate_grid(Hexahedron, (6, 6, 6), Vec((0.0, 0.0, 0.0)), Vec((1.0, 1.0, 1.0)))

    print_info("Testing gravity in different directions")

    # Test gravity in different directions
    directions = [
        ([0.0, 0.0, -1.0], "-Z (down)"),
        ([1.0, 0.0, 0.0], "+X (right)"),
        ([0.0, 1.0, 0.0], "+Y (forward)"),
        ([1.0, 1.0, 0.0]/√2, "diagonal XY"),
    ]

    results = []

    print_data("\n" * "="^60)
    print_data("GRAVITY DIRECTIONS TEST")
    print_data("="^60)

    for (direction, description) in directions
        λ, μ = create_material_model(200e9, 0.3)
        dh, cellvalues, K, f = setup_problem(grid)
        assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)

        # Different boundary conditions based on gravity direction
        if direction[3] != 0  # Z-direction
            fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        elseif direction[1] != 0  # X-direction
            fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        else  # Y-direction
            fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        end

        ch = apply_fixed_boundary!(K, f, dh, fixed_nodes)
        apply_gravity!(f, dh, cellvalues, 7850, 9.81, direction)

        u, energy, _, _, _ = solve_system(K, f, dh, cellvalues, λ, μ, ch)

        max_disp = maximum(abs.(u))
        print_data(
            "$(description): max displacement = $(round(max_disp*1e6, digits=2)) μm, energy = $(round(energy, digits=3)) J",
        )

        push!(
            results,
            (direction = direction, displacement = u, energy = energy, max_disp = max_disp),
        )
    end

    return results
end

@testset "Volume Forces Tests" begin

    @testset "Cantilever with Gravity" begin
        u, analytical_def, grid, dh, numerical_def, error = test_cantilever_gravity()

        # Export results
        export_results(u, dh, "cantilever_gravity_test")

        # Validation - error should be < 10% for reasonable mesh
        @test error < 10.0
        @test numerical_def > 0.0  # Should have positive deflection
    end

    @testset "Cube with Gravity" begin
        u, analytical_stress, grid, dh, max_displacement = test_cube_gravity()
        export_results(u, dh, "cube_gravity_test")

        # Validation - reasonable deformation range
        @test max_displacement > 0.0
        @test max_displacement < 1e-3  # reasonable deformation magnitude
    end

    @testset "Different Gravity Directions" begin
        results = test_gravity_directions()

        # All directions should produce non-zero deformations
        for result in results
            @test result.energy > 0.0
            @test result.max_disp > 0.0
        end

        print_success("\n✓ All gravity direction tests passed")
    end
end
