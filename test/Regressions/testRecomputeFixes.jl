# Regression tests for the fixes made for the dissertation recompute: node selection, surface
# node cache, SIMP material model, body force, higher-order fields, non-converged solves and
# the energy label.
using Ferrite
using LinearAlgebra
using TopOptEval.FiniteElementAnalysis: get_or_create_cache, clear_surface_cache!

"""Runs `f` with its printed output captured; returns the value of `f` and the output."""
function captured(f)
    mktemp() do path, io
        value = redirect_stdout(f, io)
        flush(io)
        return value, read(path, String)
    end
end
quiet(f) = captured(f)[1]

"""Young's modulus of the Lamé parameters returned by a material model."""
young(material, density, nu) = 2 * (1 + nu) * material(density)[2]

"""Global vector with the coordinates of each DOF: its value is the same-component coordinate."""
function dof_coordinates(dh)
    a = zeros(ndofs(dh))
    apply_analytical!(a, dh, :u, x -> x)
    return a
end

"""Resultant of a load vector: its work on the three rigid translations."""
function resultant(f, dh)
    return [
        begin
            t = zeros(ndofs(dh))
            apply_analytical!(t, dh, :u, x -> Vec{3}(i -> float(i == d)))
            dot(t, f)
        end for d = 1:3
    ]
end

"""Prismatic beam, clamped at x = 0, 1 N in -y as a uniform traction on x = L. Returns the energy."""
function beam_energy(celltype, order; L = 6.0, n = 12)
    grid = generate_grid(celltype, (n, 2, 2), Vec((0.0, 0.0, 0.0)), Vec((L, 1.0, 1.0)))
    return quiet() do
        λ, μ = create_material_model(1.0, 0.3)
        dh, cellvalues, K, f = setup_problem(grid, order)
        assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)
        fixed = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        loaded = select_nodes_by_plane(grid, [L, 0.0, 0.0], [1.0, 0.0, 0.0])
        ch = apply_fixed_boundary!(K, f, dh, fixed)
        apply_uniform_surface_traction!(
            f,
            dh,
            grid,
            get_boundary_facets(grid, loaded),
            [0.0, -1.0, 0.0],
        )
        solve_system(K, f, dh, cellvalues, λ, μ, ch)[2]
    end
end

@testset "Recompute fixes" begin
    @testset "Surface nodes are counted by global ids" begin
        for celltype in (Hexahedron, Tetrahedron)
            cache = quiet(() -> get_or_create_cache(generate_grid(celltype, (3, 3, 3))))
            @test length(cache.surface_nodes) == 4^3 - 2^3
        end
    end

    @testset "Default tolerance selects the face, not a band" begin
        grid = generate_grid(Hexahedron, (40, 8, 8), Vec((0.0, 0.0, 0.0)), Vec((10.0, 1.0, 1.0)))
        on_face = Set(n for n = 1:getnnodes(grid) if abs(grid.nodes[n].x[1]) < 1e-9)
        selected = quiet(() -> select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
        @test selected == on_face

        # An explicit tolerance still applies: the next layer of nodes, at x = 0.25
        band = quiet(() -> select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.3))
        @test on_face ⊊ band
        @test maximum(grid.nodes[n].x[1] for n in band) ≈ 0.25

        # Circle of radius 0.6 on a lattice of 0.5: the centre and its four neighbours
        grid = generate_grid(Hexahedron, (4, 4, 4), Vec((-1.0, -1.0, 0.0)), Vec((1.0, 1.0, 2.0)))
        circle = quiet(
            () -> select_nodes_by_circle(grid, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.6),
        )
        @test length(circle) == 5
        @test all(n -> grid.nodes[n].x[3] == 0.0, circle)
    end

    @testset "Surface node cache belongs to one grid" begin
        expected(n) = (n + 1)^3 - (n - 1)^3

        # A collected grid's address can be reused by the next one
        wrong = 0
        for i = 1:600
            n = 2 + mod(i, 3)
            cache = quiet(() -> get_or_create_cache(generate_grid(Hexahedron, (n, n, n))))
            wrong += length(cache.surface_nodes) != expected(n)
            i % 50 == 0 && GC.gc()
        end
        @test wrong == 0

        grid = generate_grid(Hexahedron, (2, 2, 2))
        first_cache = quiet(() -> get_or_create_cache(grid))
        @test quiet(() -> get_or_create_cache(grid)) === first_cache
        quiet(clear_surface_cache!)
        @test quiet(() -> get_or_create_cache(grid)) !== first_cache
    end

    @testset "SIMP material model" begin
        E0, nu = 2400.0, 0.35

        # Defaults of the thesis: p = 3, Emin = 1e-9 E0
        material = create_simp_material_model(E0, nu)
        @test young(material, 1.0, nu) ≈ E0
        @test young(material, 0.0, nu) ≈ 1e-9 * E0
        @test young(material, 0.5, nu) ≈ 1e-9 * E0 + (E0 - 1e-9 * E0) * 0.125

        # An explicit Emin is absolute
        @test young(create_simp_material_model(E0, nu, 1e-6, 3.0), 0.0, nu) ≈ 1e-6

        # Linear below rho_lin, continuous at it, plain SIMP above it
        Emin = 1e-9 * E0
        linear = create_simp_material_model(E0, nu; rho_lin = 0.1)
        @test young(linear, 0.05, nu) ≈ Emin + (E0 - Emin) * 0.1^2 * 0.05
        @test young(linear, 0.1 - 1e-12, nu) ≈ young(linear, 0.1, nu)
        @test young(linear, 0.5, nu) ≈ young(material, 0.5, nu)
    end

    @testset "Body force on every cell with non-zero density" begin
        grid = generate_grid(Hexahedron, (4, 4, 4), Vec((0.0, 0.0, 0.0)), Vec((4.0, 4.0, 4.0)))
        dh, cellvalues, _, _ = quiet(() -> setup_problem(grid))

        # Effective density of 1e-9 t/mm³ on half of the cells, void on the others
        density = [isodd(c) ? 1.04e-9 : 0.0 for c = 1:getncells(grid)]
        f = zeros(ndofs(dh))
        quiet(
            () -> apply_variable_density_volume_force!(
                f,
                dh,
                cellvalues,
                [0.0, 6000.0, 0.0],
                density,
            ),
        )
        @test resultant(f, dh) ≈ [0.0, 6000.0 * sum(density), 0.0] atol = 1e-15
    end

    @testset "Higher-order fields" begin
        for celltype in (Tetrahedron, Hexahedron)
            grid = generate_grid(celltype, (3, 2, 2), Vec((0.0, 0.0, 0.0)), Vec((3.0, 2.0, 2.0)))
            dh, cellvalues, K, f = quiet(() -> setup_problem(grid, 2))
            coordinates = dof_coordinates(dh)

            # The DOFs of a node hold the components of its own coordinates
            node_dofs = get_node_dofs(dh)
            @test length(node_dofs) == getnnodes(grid)
            @test all(coordinates[node_dofs[n]] ≈ grid.nodes[n].x for n = 1:getnnodes(grid))

            # The load keeps its direction and magnitude
            loaded = Set(n for n = 1:getnnodes(grid) if grid.nodes[n].x[1] == 3.0)
            fp = zeros(ndofs(dh))
            quiet(() -> apply_force!(fp, dh, loaded, [1.0, -2.0, 3.0]))
            @test resultant(fp, dh) ≈ [1.0, -2.0, 3.0]

            ft = zeros(ndofs(dh))
            quiet(
                () -> apply_uniform_surface_traction!(
                    ft,
                    dh,
                    grid,
                    get_boundary_facets(grid, loaded),
                    [1.0, -2.0, 3.0],
                ),
            )
            @test resultant(ft, dh) ≈ [1.0, -2.0, 3.0]

            # A clamped face holds every DOF on it, not only those on the vertices
            fixed = Set(n for n = 1:getnnodes(grid) if grid.nodes[n].x[1] == 0.0)
            ch = quiet(() -> apply_fixed_boundary!(K, f, dh, fixed))
            # (the DOFs of a point are consecutive, starting with its x component)
            on_face = Set{Int}()
            for cell in CellIterator(dh)
                cell_dofs = celldofs(cell)
                for i = 1:3:length(cell_dofs)
                    iszero(coordinates[cell_dofs[i]]) && union!(on_face, cell_dofs[i:(i+2)])
                end
            end
            @test Set(ch.prescribed_dofs) == on_face

            # A sliding face holds only its component
            sl = quiet(() -> apply_sliding_boundary!(K, f, dh, fixed, [1]))
            @test length(sl.prescribed_dofs) == length(on_face) ÷ 3
        end

        # P1 keeps constraining exactly the three DOFs of each node
        grid = generate_grid(Tetrahedron, (3, 2, 2), Vec((0.0, 0.0, 0.0)), Vec((3.0, 2.0, 2.0)))
        dh, _, K, f = quiet(() -> setup_problem(grid, 1))
        fixed = Set(n for n = 1:getnnodes(grid) if grid.nodes[n].x[1] == 0.0)
        ch = quiet(() -> apply_fixed_boundary!(K, f, dh, fixed))
        @test length(ch.prescribed_dofs) == 3 * length(fixed)

        # Beam against the bending + shear solution: U = F δ / 2, with δ = FL³/3EI + FL/κGA
        L, E, G, I, A, κ = 6.0, 1.0, 1.0 / 2.6, 1.0 / 12.0, 1.0, 5.0 / 6.0
        analytical = 0.5 * (L^3 / (3 * E * I) + L / (κ * G * A))
        P1 = beam_energy(Tetrahedron, 1)
        for celltype in (Tetrahedron, Hexahedron)
            P2 = beam_energy(celltype, 2)
            @test isapprox(P2, analytical; rtol = 0.03)
            @test P2 > P1
        end

        # The result of a P2 field can be written out
        mktempdir() do dir
            grid = generate_grid(Tetrahedron, (2, 1, 1))
            dh, _, _, _ = quiet(() -> setup_problem(grid, 2))
            path = joinpath(dir, "p2")
            quiet(() -> export_results(zeros(ndofs(dh)), dh, path))
            @test isfile(path * ".vtu")
        end
    end

    @testset "Non-converged iterative solve" begin
        function system()
            grid = generate_grid(Hexahedron, (8, 2, 2), Vec((0.0, 0.0, 0.0)), Vec((8.0, 2.0, 2.0)))
            return quiet() do
                λ, μ = create_material_model(1.0, 0.3)
                dh, cellvalues, K, f = setup_problem(grid)
                assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)
                fixed = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
                loaded = select_nodes_by_plane(grid, [8.0, 0.0, 0.0], [1.0, 0.0, 0.0])
                ch = apply_fixed_boundary!(K, f, dh, fixed)
                apply_force!(f, dh, loaded, [0.0, -1.0, 0.0])
                (K, f, dh, cellvalues, λ, μ, ch)
            end
        end
        settings = (method = :cg, preconditioner = :diagonal, verbose = false)

        capped = SolverConfig(; settings..., max_iterations = 2)
        @test_throws ErrorException quiet(
            () -> solve_system_robust(system()...; config = capped),
        )

        accepted = SolverConfig(; settings..., max_iterations = 2, allow_unconverged = true)
        @test_logs (:warn, r"did not converge") match_mode = :any quiet(
            () -> solve_system_robust(system()...; config = accepted),
        )

        direct = quiet(() -> solve_system(system()...))[2]
        converged = quiet(
            () -> solve_system_robust(system()...; config = SolverConfig(; settings...)),
        )[2]
        @test converged ≈ direct rtol = 1e-6
    end

    @testset "Energy is not labelled as joules" begin
        grid = generate_grid(Hexahedron, (2, 1, 1))
        _, output = captured() do
            λ, μ = create_material_model(1.0, 0.3)
            dh, cellvalues, K, f = setup_problem(grid)
            assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)
            fixed = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            ch = apply_fixed_boundary!(K, f, dh, fixed)
            apply_force!(f, dh, fixed, [0.0, 0.0, 0.0])
            solve_system(K, f, dh, cellvalues, λ, μ, ch)
        end
        @test occursin("Deformation energy", output)
        @test occursin("mJ for N-mm-MPa", output)
    end
end
