# =============================================================================
# 3D BEAM COMPLIANCE EVALUATION - 4-CORNER FIXATION
# =============================================================================
#
# Geometry: 2.0 × 1.0 × 1.0 (X × Y × Z)
# BC:       4 corners fixed at x=0 (0.3 × 0.3 each)
# Load:     Circular region r=0.1 at x=2.0 center, F=[0,0,-1] via surface traction
#
# Scans directory for *_TET.vtu files and evaluates compliance.
#
# =============================================================================

using Test
using LinearAlgebra
using Printf
using Dates
using TopOptEval
using TopOptEval.Utils
using Ferrite

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR = @__DIR__

# Geometry
const XMAX = 2.0
const YMAX = 1.0
const ZMAX = 1.0
const FIX_SIZE = 0.3
const LOAD_RADIUS = 0.1

# Material
const E0 = 1.0
const NU = 0.3

# Loading
const TOTAL_FORCE = [0.0, 0.0, -1.0]

# Tolerances (consistent with EasySIMP)
const PLANE_TOL = 1e-6
const GEOM_TOL = 1e-6

# =============================================================================
# NODE SELECTION
# =============================================================================

"""
Scan directory for *_TET.vtu files.
"""
function find_tet_vtu_files(directory::String)
    !isdir(directory) && error("Directory not found: $directory")
    files = [
        joinpath(directory, f) for
        f in readdir(directory) if endswith(lowercase(f), "_tet.vtu")
    ]
    sort!(files)
end

"""
Select nodes at 4 corners of x=0 face (FIX_SIZE × FIX_SIZE regions).
"""
function select_4corner_fixed_nodes(grid)
    fixed = Set{Int}()
    for nid = 1:getnnodes(grid)
        x, y, z = grid.nodes[nid].x
        if abs(x) < PLANE_TOL
            c1 = (y <= FIX_SIZE + GEOM_TOL) && (z <= FIX_SIZE + GEOM_TOL)
            c2 = (y >= YMAX - FIX_SIZE - GEOM_TOL) && (z <= FIX_SIZE + GEOM_TOL)
            c3 = (y <= FIX_SIZE + GEOM_TOL) && (z >= ZMAX - FIX_SIZE - GEOM_TOL)
            c4 = (y >= YMAX - FIX_SIZE - GEOM_TOL) && (z >= ZMAX - FIX_SIZE - GEOM_TOL)
            (c1 || c2 || c3 || c4) && push!(fixed, nid)
        end
    end
    fixed
end

"""
Select nodes in circular region at x=XMAX face center.
"""
function select_circular_force_nodes(grid)
    nodes = Set{Int}()
    cy, cz = YMAX / 2, ZMAX / 2
    for nid = 1:getnnodes(grid)
        x, y, z = grid.nodes[nid].x
        if abs(x - XMAX) < PLANE_TOL
            dist_sq = (y - cy)^2 + (z - cz)^2
            (dist_sq <= LOAD_RADIUS^2 + GEOM_TOL) && push!(nodes, nid)
        end
    end
    isempty(nodes) && error("No force nodes found in circular region at x=$XMAX")
    nodes
end

# =============================================================================
# ANALYSIS
# =============================================================================

"""
Export results summary to text file.
"""
function export_results_txt(name, compliance, volume, area, dir)
    path = joinpath(dir, "$(name).txt")
    open(path, "w") do io
        println(io, "=" ^ 55)
        println(io, "  COMPLIANCE ANALYSIS (Surface Traction)")
        println(io, "=" ^ 55)
        println(io, "  Task:          $name")
        println(io, "  Compliance:    $(@sprintf("%.6f", compliance))")
        println(io, "  Volume:        $(@sprintf("%.6f", volume))")
        println(io, "  Boundary area: $(@sprintf("%.6f", area))")
        println(io, "  Traction:      $(@sprintf("%.4f", norm(TOTAL_FORCE)/area)) N/unit²")
        println(io, "  Generated:     $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "=" ^ 55)
    end
    path
end

"""
Analyze tetrahedral mesh using surface traction loading.
"""
function analyze_tet_mesh(filepath::String, taskname::String, output_dir::String)
    print_info("\n" * "=" ^ 65)
    print_info("ANALYZING: $taskname")
    print_info("=" ^ 65)

    # Import and setup
    grid = import_mesh(filepath)
    volume = Utils.calculate_volume(grid)
    λ, μ = create_material_model(E0, NU)
    dh, cellvalues, K, f = setup_problem(grid)

    print_info(
        "Elements: $(getncells(grid)), Nodes: $(getnnodes(grid)), DOFs: $(ndofs(dh))",
    )

    # Assemble
    assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)

    # Boundary conditions
    fixed_nodes = select_4corner_fixed_nodes(grid)
    force_nodes = select_circular_force_nodes(grid)
    force_facets = get_boundary_facets(grid, force_nodes)
    boundary_area = compute_boundary_area(grid, dh, force_facets)

    print_info("Fixed nodes: $(length(fixed_nodes)), Force facets: $(length(force_facets))")
    print_info("Boundary area: $(round(boundary_area, digits=6))")

    # Apply BC and load
    ch = apply_fixed_boundary!(K, f, dh, fixed_nodes)
    apply_uniform_surface_traction!(f, dh, grid, force_facets, TOTAL_FORCE)

    # Solve
    u, energy, stress_field, max_vm, max_cell = solve_system(K, f, dh, cellvalues, λ, μ, ch)
    compliance = dot(f, u)

    # Results
    print_success("Compliance: $compliance")
    print_data("  Energy: $energy J, Max vM: $max_vm")

    export_results_txt(taskname, compliance, volume, boundary_area, output_dir)

    (
        name = taskname,
        compliance = compliance,
        energy = energy,
        volume = volume,
        boundary_area = boundary_area,
        max_von_mises = max_vm,
        n_elements = getncells(grid),
        n_nodes = getnnodes(grid),
        n_dofs = ndofs(dh),
    )
end

# =============================================================================
# MAIN
# =============================================================================

@testset "3D Beam 4-Corner (Surface Traction)" begin
    print_info("Directory: $WORK_DIR")
    files = find_tet_vtu_files(WORK_DIR)

    isempty(files) && (print_warning("No *_TET.vtu files found"); @test false)
    print_info("Found $(length(files)) files\n")

    results = []
    for f in files
        name = splitext(basename(f))[1]
        @testset "$name" begin
            try
                r = analyze_tet_mesh(f, name, dirname(f))
                push!(results, r)
                @test r.compliance > 0.0
            catch e
                print_error("Failed: $e")
                @test false
            end
        end
    end

    # Summary
    if !isempty(results)
        println("\n" * "=" ^ 85)
        println("SUMMARY")
        println("=" ^ 85)
        println("┌" * "─"^40 * "┬" * "─"^14 * "┬" * "─"^14 * "┬" * "─"^12 * "┐")
        println(
            "│ " *
            rpad("Mesh", 38) *
            " │ " *
            lpad("Compliance", 12) *
            " │ " *
            lpad("Area", 12) *
            " │ " *
            lpad("Elements", 10) *
            " │",
        )
        println("├" * "─"^40 * "┼" * "─"^14 * "┼" * "─"^14 * "┼" * "─"^12 * "┤")

        ref = results[1].compliance
        for r in results
            n = length(r.name) > 38 ? r.name[1:35] * "..." : r.name
            println(
                "│ " *
                rpad(n, 38) *
                " │ " *
                lpad(@sprintf("%.6f", r.compliance), 12) *
                " │ " *
                lpad(@sprintf("%.6f", r.boundary_area), 12) *
                " │ " *
                lpad(string(r.n_elements), 10) *
                " │",
            )
        end
        println("└" * "─"^40 * "┴" * "─"^14 * "┴" * "─"^14 * "┴" * "─"^12 * "┘")

        println("\nRelative to $(results[1].name):")
        for r in results
            diff = (r.compliance - ref) / ref * 100
            println("  $(r.name): $(diff >= 0 ? "+" : "")$(round(diff, digits=2))%")
        end
        println("=" ^ 85)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("\n3D BEAM 4-CORNER FIXATION")
    println("Domain: $XMAX × $YMAX × $ZMAX, Fixed corners: $(FIX_SIZE)×$(FIX_SIZE)")
    println("Load: r=$LOAD_RADIUS at x=$XMAX, F=$TOTAL_FORCE (surface traction)")
    println("Files: ", length(find_tet_vtu_files(WORK_DIR)))
end
