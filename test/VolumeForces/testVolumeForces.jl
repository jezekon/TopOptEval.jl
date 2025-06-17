using Ferrite
"""
Testování gravitace na konzolovém nosníku - máme analytické řešení!
"""
function test_cantilever_gravity()
    # Geometrie nosníku
    L, w, h = 10.0, 1.0, 1.0  # délka, šířka, výška
    grid = generate_grid(Hexahedron, (20, 4, 4), Vec((0.0, 0.0, 0.0)), Vec((L, w, h)))
    
    # Material properties
    E = 200e9  # Pa (ocel)
    ν = 0.3
    ρ = 7850   # kg/m³ (hustota oceli)
    g = 9.81   # m/s²
    
    # Setup FE problem
    λ, μ = create_material_model(E, ν)
    dh, cellvalues, K, f = setup_problem(grid)
    
    # Assemble stiffness matrix
    assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)
    
    # Apply boundary conditions - vetknutí na x=0
    fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ch = apply_fixed_boundary!(K, f, dh, fixed_nodes)
    
    # Apply gravity as volume force
    apply_gravity!(f, dh, cellvalues, ρ, g, [0.0, 0.0, -1.0])
    
    # Solve
    u, energy, stress_field, max_vm, max_cell = solve_system(K, f, dh, cellvalues, λ, μ, ch)
    
    # Analytické řešení pro maximální průhyb konzoly:
    # δ_max = (ρ * g * L⁴) / (8 * E * I)
    I = w * h³ / 12  # moment setrvačnosti
    analytical_deflection = (ρ * g * L^4) / (8 * E * I)
    
    println("Analytický maximální průhyb: ", analytical_deflection, " m")
    
    return u, analytical_deflection, grid, dh
end

"""
Test gravitace na jednoduché kostce - uniform stress field
"""
function test_cube_gravity()
    # Jednotková kostka
    grid = generate_grid(Hexahedron, (8, 8, 8), Vec((0.0, 0.0, 0.0)), Vec((1.0, 1.0, 1.0)))
    
    # Material properties
    E = 200e9
    ν = 0.3
    ρ = 7850
    
    λ, μ = create_material_model(E, ν)
    dh, cellvalues, K, f = setup_problem(grid)
    assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)
    
    # Vetknutí spodní plochy (z=0)
    fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    ch = apply_fixed_boundary!(K, f, dh, fixed_nodes)
    
    # Apply gravity
    apply_gravity!(f, dh, cellvalues, ρ)
    
    # Solve
    u, energy, stress_field, max_vm, max_cell = solve_system(K, f, dh, cellvalues, λ, μ, ch)
    
    # Analytické napětí: σ_z = ρ * g * h na spodku
    analytical_stress = ρ * 9.81 * 1.0  # Pa
    println("Analytické napětí na spodku: ", analytical_stress, " Pa")
    
    return u, analytical_stress, grid, dh
end

"""
Test různých směrů gravitace pro validaci
"""
function test_gravity_directions()
    grid = generate_grid(Hexahedron, (6, 6, 6), Vec((0.0, 0.0, 0.0)), Vec((1.0, 1.0, 1.0)))
    
    # Test gravitace v různých směrech
    directions = [
        [0.0, 0.0, -1.0],  # -Z (dolů)
        [1.0, 0.0, 0.0],   # +X (doprava)
        [0.0, 1.0, 0.0],   # +Y (vpřed)
        [1.0, 1.0, 0.0]/√2 # diagonálně
    ]
    
    results = []
    
    for (i, direction) in enumerate(directions)
        println("Testing gravity direction: ", direction)
        
        λ, μ = create_material_model(200e9, 0.3)
        dh, cellvalues, K, f = setup_problem(grid)
        assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)
        
        # Různé vetknutí podle směru gravitace
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
        
        push!(results, (direction=direction, displacement=u, energy=energy))
    end
    
    return results
end

# using Test
# using TopOptEval
# using LinearAlgebra

@testset "Volume Forces Tests" begin
    
    @testset "Cantilever with Gravity" begin
        u, analytical_def, grid, dh = test_cantilever_gravity()
        
        # Export results
        export_results(u, dh, "cantilever_gravity_test")
        
        # Porovnání s analytickým řešením
        # Najdeme maximální průhyb v Z směru
        max_deflection = maximum(abs.(u[3:3:end]))  # každý třetí DOF je Z
        
        println("Numerical max deflection: ", max_deflection)
        println("Analytical deflection: ", analytical_def)
        println("Relative error: ", abs(max_deflection - analytical_def) / analytical_def * 100, "%")
        
        # Test by měl dát error < 5% pro dostatečně jemnou síť
        @test abs(max_deflection - analytical_def) / analytical_def < 0.1
    end
    
    @testset "Cube with Gravity" begin
        u, analytical_stress, grid, dh = test_cube_gravity()
        export_results(u, dh, "cube_gravity_test")
        
        # Kontrola zda je deformace rozumná
        max_displacement = maximum(abs.(u))
        @test max_displacement > 0.0
        @test max_displacement < 1e-3  # rozumná deformace
    end
    
    @testset "Different Gravity Directions" begin
        results = test_gravity_directions()
        
        # Všechny směry by měly dát nenulové deformace
        for result in results
            @test result.energy > 0.0
            @test maximum(abs.(result.displacement)) > 0.0
        end
    end
end
