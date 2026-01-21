# =============================================================================
# 3D MBB BEAM COMPLIANCE EVALUATION
# =============================================================================
#
# Geometry: 2.0 × 1.0 × 1.0 (X × Y × Z)
# BC:       Symmetry at x=0 (U1=0), Support at x=2,y=0 edge (U2=0)
# Load:     Semicircle r=0.1 at (0, 1, 0.5) on top face, F=[0,-1,0] via surface traction
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
const SUPPORT_WIDTH = 0.05
const LOAD_RADIUS = 0.1

# Material
const E0 = 1.0
const NU = 0.3

# Loading
const TOTAL_FORCE = [0.0, -1.0, 0.0]

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
Select nodes on x=0 face for symmetry BC (U1=0).
"""
function select_symmetry_nodes(grid)
    nodes = Set{Int}()
    for nid = 1:getnnodes(grid)
        x = grid.nodes[nid].x[1]
        abs(x) < PLANE_TOL && push!(nodes, nid)
    end
    nodes
end

"""
Select nodes on right bottom edge (x >= XMAX-SUPPORT_WIDTH, y=0) for support BC (U2=0).
"""
function select_support_nodes(grid)
    nodes = Set{Int}()
    for nid = 1:getnnodes(grid)
        x, y = grid.nodes[nid].x[1], grid.nodes[nid].x[2]
        if abs(y) < PLANE_TOL && x >= XMAX - SUPPORT_WIDTH - GEOM_TOL
            push!(nodes, nid)
        end
    end
    nodes
end

"""
Select nodes in semicircular region on top face (y=YMAX).
Center at (0, YMAX, ZMAX/2), x >= 0 (right half of circle).
"""
function select_semicircle_force_nodes(grid)
    nodes = Set{Int}()
    cx, cz = 0.0, ZMAX / 2

    for nid = 1:getnnodes(grid)
        x, y, z = grid.nodes[nid].x
        if abs(y - YMAX) < PLANE_TOL
            dist_sq = (x - cx)^2 + (z - cz)^2
            # Semicircle: within radius AND x >= 0
            if dist_sq <= LOAD_RADIUS^2 + GEOM_TOL && x >= cx - GEOM_TOL
                push!(nodes, nid)
            end
        end
    end
    isempty(nodes) && error("No force nodes found in semicircular region at y=$YMAX")
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
        println(io, "  MBB COMPLIANCE ANALYSIS (Surface Traction)")
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

    # Boundary nodes
    symmetry_nodes = select_symmetry_nodes(grid)
    support_nodes = select_support_nodes(grid)
    force_nodes = select_semicircle_force_nodes(grid)

    # Get facets for surface traction
    force_facets = get_boundary_facets(grid, force_nodes)
    boundary_area = compute_boundary_area(grid, dh, force_facets)

    print_info(
        "Symmetry nodes: $(length(symmetry_nodes)), Support nodes: $(length(support_nodes))",
    )
    print_info(
        "Force facets: $(length(force_facets)), Boundary area: $(round(boundary_area, digits=6))",
    )

    # Apply sliding BCs
    ch1 = apply_sliding_boundary!(K, f, dh, symmetry_nodes, [1])  # Fix X only
    ch2 = apply_sliding_boundary!(K, f, dh, support_nodes, [2])   # Fix Y only

    # Apply surface traction
    apply_uniform_surface_traction!(f, dh, grid, force_facets, TOTAL_FORCE)

    # Solve (pass both constraint handlers)
    u, energy, stress_field, max_vm, max_cell =
        solve_system(K, f, dh, cellvalues, λ, μ, ch1, ch2)
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

@testset "3D MBB Beam (Surface Traction)" begin
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
    println("\n3D MBB BEAM")
    println("Domain: $XMAX × $YMAX × $ZMAX")
    println("Symmetry: x=0 (U1=0), Support: x≥$(XMAX-SUPPORT_WIDTH),y=0 (U2=0)")
    println("Load: semicircle r=$LOAD_RADIUS at (0,$YMAX,$(ZMAX/2)), F=$TOTAL_FORCE")
    println("Files: ", length(find_tet_vtu_files(WORK_DIR)))
end
