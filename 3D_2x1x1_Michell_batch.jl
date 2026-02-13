# =============================================================================
# 3D MICHELL-TYPE BEAM COMPLIANCE EVALUATION
# =============================================================================
#
# Geometry: 2.0 × 1.0 × 1.0 (X × Y × Z)
# BC:       4 corner supports on bottom face (y=0), 0.15×0.15 each, U1=U2=U3=0
# Load:     Circle r=0.1 at (1, 0, 0.5) on bottom face, F=[0,-1,0] via surface traction
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
const CORNER_SIZE = 0.15
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
Select nodes in 4 corner regions on bottom face (y=0) for fixed support (U1=U2=U3=0).
Corners: (x≈0,z≈0), (x≈0,z≈zmax), (x≈xmax,z≈0), (x≈xmax,z≈zmax)
Each corner region is CORNER_SIZE × CORNER_SIZE.
"""
function select_support_left_nodes(grid)
    nodes = Set{Int}()
    for nid = 1:getnnodes(grid)
        coord = grid.nodes[nid].x
        if abs(coord[2]) < PLANE_TOL && coord[1] <= CORNER_SIZE + GEOM_TOL
            # Bottom-left corner (x≈0, z≈0)
            in_corner1 = coord[3] <= CORNER_SIZE + GEOM_TOL
            # Top-left corner (x≈0, z≈zmax)
            in_corner2 = coord[3] >= ZMAX - CORNER_SIZE - GEOM_TOL
            if in_corner1 || in_corner2
                push!(nodes, nid)
            end
        end
    end
    nodes
end

function select_support_right_nodes(grid)
    nodes = Set{Int}()
    for nid = 1:getnnodes(grid)
        coord = grid.nodes[nid].x
        if abs(coord[2]) < PLANE_TOL && coord[1] >= XMAX - CORNER_SIZE - GEOM_TOL
            # Bottom-right corner (x≈xmax, z≈0)
            in_corner1 = coord[3] <= CORNER_SIZE + GEOM_TOL
            # Top-right corner (x≈xmax, z≈zmax)
            in_corner2 = coord[3] >= ZMAX - CORNER_SIZE - GEOM_TOL
            if in_corner1 || in_corner2
                push!(nodes, nid)
            end
        end
    end
    nodes
end

"""
Select nodes in circular region on bottom face (y=0).
Center at (1.0, 0.0, 0.5), radius LOAD_RADIUS.
"""
function select_circle_force_nodes(grid)
    nodes = Set{Int}()
    cx, cz = 1.0, ZMAX / 2

    for nid = 1:getnnodes(grid)
        coord = grid.nodes[nid].x
        if abs(coord[2]) < PLANE_TOL
            dx = coord[1] - cx
            dz = coord[3] - cz
            dist = sqrt(dx^2 + dz^2)
            if dist <= LOAD_RADIUS + GEOM_TOL
                push!(nodes, nid)
            end
        end
    end
    isempty(nodes) &&
        error("No force nodes found in circular region at y=0, center=($cx, 0, $cz)")
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
        println(io, "  MICHELL-TYPE COMPLIANCE ANALYSIS (Surface Traction)")
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

    # Boundary nodes - 4 corner supports (left pair + right pair)
    support_left = select_support_left_nodes(grid)
    support_right = select_support_right_nodes(grid)
    force_nodes = select_circle_force_nodes(grid)

    # Get facets for surface traction
    force_facets = get_boundary_facets(grid, force_nodes)
    boundary_area = compute_boundary_area(grid, dh, force_facets)

    print_info(
        "Support left: $(length(support_left)), Support right: $(length(support_right))",
    )
    print_info(
        "Force facets: $(length(force_facets)), Boundary area: $(round(boundary_area, digits=6))",
    )

    # Apply fixed BCs (U1=U2=U3=0) on both corner groups
    ch1 = apply_fixed_boundary!(K, f, dh, support_left)
    ch2 = apply_fixed_boundary!(K, f, dh, support_right)

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

@testset "3D Michell-Type Beam (Surface Traction)" begin
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
    println("\n3D MICHELL-TYPE BEAM")
    println("Domain: $XMAX × $YMAX × $ZMAX")
    println(
        "Support: 4 corners on y=0, $(CORNER_SIZE)×$(CORNER_SIZE) each, fixed U1=U2=U3=0",
    )
    println("Load: circle r=$LOAD_RADIUS at (1,0,$(ZMAX/2)), F=$TOTAL_FORCE")
    println("Files: ", length(find_tet_vtu_files(WORK_DIR)))
end
