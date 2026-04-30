using Test
using LinearAlgebra
using TopOptEval
using TopOptEval.Utils
using Ferrite

"""
Select nodes at a coordinate plane (axis = 1,2,3) by direct iteration.
"""
function nodes_at_plane(grid, axis::Int, value::Float64; tol = 1e-6)
    nodes = Set{Int}()
    for nid = 1:getnnodes(grid)
        if abs(grid.nodes[nid].x[axis] - value) < tol
            push!(nodes, nid)
        end
    end
    nodes
end

@testset "TopOptEval.jl" begin
    @testset "Linear cantilever beam" begin
        taskName = "cantilever_beam-linear"

        grid = import_mesh(joinpath(@__DIR__, "..", "data", "beam_linear_volume_mesh.vtu"))
        volume = calculate_volume(grid)
        @test volume > 0.0

        λ, μ = create_material_model(1.0, 0.3)
        dh, cellvalues, K, f = setup_problem(grid)
        assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)

        fixed_nodes = nodes_at_plane(grid, 1, 0.0)
        load_nodes = nodes_at_plane(grid, 1, 60.0)
        @test !isempty(fixed_nodes)
        @test !isempty(load_nodes)

        ch = apply_fixed_boundary!(K, f, dh, fixed_nodes)
        apply_force!(f, dh, collect(load_nodes), [0.0, 0.0, -1.0])

        u, energy, stress_field, max_von_mises, max_stress_cell =
            solve_system(K, f, dh, cellvalues, λ, μ, ch)

        @test energy > 0.0
        @test max_von_mises > 0.0
        @test all(isfinite, u)

        export_results(u, dh, "$(taskName)_u")
        export_results(stress_field, dh, "$(taskName)_stress")
    end

    @testset "SIMP beam (raw density)" begin
        taskName = "cantilever_beam-raw"

        mesh_path = joinpath(@__DIR__, "..", "data", "beam_vfrac_04_Raw.vtu")
        grid = import_mesh(mesh_path)
        density_data = extract_cell_density(mesh_path)
        @test length(density_data) == getncells(grid)

        volume = Utils.calculate_volume(grid, density_data)
        @test volume > 0.0

        # SIMP material model
        E0 = 1.0
        nu = 0.3
        Emin = 1e-8
        p = 3.0
        material_model = create_simp_material_model(E0, nu, Emin, p)

        dh, cellvalues, K, f = setup_problem(grid)
        assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, density_data)

        fixed_nodes = nodes_at_plane(grid, 1, 0.0)
        load_nodes = nodes_at_plane(grid, 1, 60.0)
        @test !isempty(fixed_nodes)
        @test !isempty(load_nodes)

        ch = apply_fixed_boundary!(K, f, dh, fixed_nodes)
        apply_force!(f, dh, collect(load_nodes), [0.0, 0.0, -1.0])

        u, energy, stress_field, max_von_mises, max_stress_cell =
            solve_system_simp(K, f, dh, cellvalues, material_model, density_data, ch)

        @test energy > 0.0
        @test max_von_mises > 0.0
        @test all(isfinite, u)

        export_results(u, dh, "$(taskName)_u")
        export_results(stress_field, dh, "$(taskName)_stress")
    end
end
