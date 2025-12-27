# =============================================================================
# 3D MBB BEAM COMPLIANCE EVALUATION SCRIPT
# =============================================================================
#
# Description:
#   Evaluates compliance (deformation energy) for all tetrahedral mesh files
#   in the current directory. Automatically scans for *_TET.vtu files and
#   processes each one with MBB boundary conditions.
#
# Geometry:
#   - Domain: 2.0 × 1.0 × 1.0 (X × Y × Z)
#   - Symmetry: Left face (x=0) - U1=0 only
#   - Support: Right bottom edge (x=2, y=0, all Z) - U2=0 only
#   - Load: Semicircle on top face at (0, 1, 0.5), radius 0.1, force in -Y
#
# Usage:
#   Place this script in the directory with *_TET.vtu files and run it.
#   Results (.txt files) will be saved in the same directory.
#
# =============================================================================

using Test
using LinearAlgebra
using Printf
using Dates
using TopOptEval
using TopOptEval.Utils
using Ferrite  # For getnnodes, getncells, ndofs

# =============================================================================
# CONFIGURATION
# =============================================================================

# Working directory = directory where this script is located
WORK_DIR = @__DIR__

# Geometry parameters
XMAX = 2.0
YMAX = 1.0
ZMAX = 1.0
SUPPORT_WIDTH = 0.05     # Width of support region at right edge (x >= XMAX - SUPPORT_WIDTH)
LOAD_RADIUS = 0.1        # Radius of semicircular load region

# Material properties
E0 = 1.0                 # Young's modulus
NU = 0.3                 # Poisson's ratio

# Force vector (MBB: force in -Y direction)
FORCE_VECTOR = [0.0, -1.0, 0.0]

# Tolerance for node selection (scaled for 2x1x1 domain)
NODE_SELECTION_TOL = 0.005

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

"""
    find_tet_vtu_files(directory::String)

Scans the specified directory for all *_TET.vtu files (tetrahedral meshes only).

Returns:
- Vector{String}: List of full paths to TET .vtu files
"""
function find_tet_vtu_files(directory::String)
    if !isdir(directory)
        error("Directory not found: $directory")
    end

    vtu_files = String[]
    for file in readdir(directory)
        # Only process files ending with _TET.vtu (case insensitive)
        if endswith(lowercase(file), "_tet.vtu")
            push!(vtu_files, joinpath(directory, file))
        end
    end

    # Sort alphabetically for consistent ordering
    sort!(vtu_files)

    return vtu_files
end

"""
    select_symmetry_nodes(grid)

Selects nodes on the left face (x=0) for symmetry boundary condition.
These nodes will be fixed in X direction only (U1=0).

Returns:
- Set{Int}: Set of node IDs for symmetry boundary condition
"""
function select_symmetry_nodes(grid)
    symmetry_nodes = Set{Int}()

    for node_id = 1:getnnodes(grid)
        coord = grid.nodes[node_id].x
        x = coord[1]

        # Check if node is on the x=0 face (symmetry plane)
        if abs(x) < NODE_SELECTION_TOL
            push!(symmetry_nodes, node_id)
        end
    end

    return symmetry_nodes
end

"""
    select_support_nodes(grid)

Selects nodes on the right bottom edge for support boundary condition.
These nodes are at y=0 (bottom face) and x >= XMAX - SUPPORT_WIDTH (right edge).
They will be fixed in Y direction only (U2=0).

Returns:
- Set{Int}: Set of node IDs for support boundary condition
"""
function select_support_nodes(grid)
    support_nodes = Set{Int}()

    for node_id = 1:getnnodes(grid)
        coord = grid.nodes[node_id].x
        x, y = coord[1], coord[2]

        # Check if node is on bottom face (y=0) AND on right edge (x >= XMAX - SUPPORT_WIDTH)
        if abs(y) < NODE_SELECTION_TOL && x >= XMAX - SUPPORT_WIDTH - NODE_SELECTION_TOL
            push!(support_nodes, node_id)
        end
    end

    return support_nodes
end

"""
    select_semicircle_force_nodes(grid)

Selects nodes in a semicircular region on the top face (y=YMAX).
The semicircle is centered at (0, YMAX, ZMAX/2) with x >= 0.

Returns:
- Set{Int}: Set of node IDs for force application
"""
function select_semicircle_force_nodes(grid)
    force_nodes = Set{Int}()

    # Force center on top face (y=YMAX), at x=0, z=ZMAX/2
    force_center_x = 0.0
    force_center_z = ZMAX / 2

    for node_id = 1:getnnodes(grid)
        coord = grid.nodes[node_id].x
        x, y, z = coord[1], coord[2], coord[3]

        # Check if node is on the top face (y=YMAX)
        if abs(y - YMAX) < NODE_SELECTION_TOL
            # Calculate distance from center in XZ plane
            dx = x - force_center_x
            dz = z - force_center_z
            dist = sqrt(dx^2 + dz^2)

            # Semicircle condition: within radius AND x >= 0 (right half)
            if dist <= LOAD_RADIUS + NODE_SELECTION_TOL &&
               x >= force_center_x - NODE_SELECTION_TOL
                push!(force_nodes, node_id)
            end
        end
    end

    # Fallback: if no nodes found, find closest node to force center
    if isempty(force_nodes)
        force_center = [force_center_x, YMAX, force_center_z]
        min_dist = Inf
        closest_node = 1

        for node_id = 1:getnnodes(grid)
            coord = grid.nodes[node_id].x
            dist = norm([coord[1], coord[2], coord[3]] - force_center)
            if dist < min_dist
                min_dist = dist
                closest_node = node_id
            end
        end

        push!(force_nodes, closest_node)
        @warn "No nodes found in semicircular region, using closest node: $closest_node"
    end

    return force_nodes
end

"""
    export_results_txt(taskname::String, compliance::Float64, volume::Float64, output_dir::String)

Exports analysis results to a .txt summary file.

Parameters:
- taskname: Name of the task/mesh
- compliance: Calculated compliance value
- volume: Mesh volume
- output_dir: Directory for output file
"""
function export_results_txt(
    taskname::String,
    compliance::Float64,
    volume::Float64,
    output_dir::String,
)
    txt_path = joinpath(output_dir, "$(taskname).txt")

    open(txt_path, "w") do io
        println(io, "==================================================")
        println(io, "  TOPOLOGY OPTIMIZATION COMPLIANCE ANALYSIS")
        println(io, "==================================================")
        println(io, "  Task name:    $taskname")
        println(io, "  Compliance:   $(@sprintf("%.6f", compliance))")
        println(io, "  Volume:       $(@sprintf("%.6f", volume))")
        println(io, "  Generated:    $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "==================================================")
    end

    return txt_path
end

"""
    analyze_tet_mesh(filepath::String, taskname::String, output_dir::String)

Analyzes a solid tetrahedral mesh.

Parameters:
- filepath: Path to the VTU file
- taskname: Name for output files
- output_dir: Directory for output files

Returns:
- NamedTuple with analysis results
"""
function analyze_tet_mesh(filepath::String, taskname::String, output_dir::String)
    print_info("\n" * "="^70)
    print_info("ANALYZING TET MESH: $taskname")
    print_info("="^70)

    # 1. Import mesh
    print_info("Importing mesh from: $filepath")
    grid = import_mesh(filepath)

    # 2. Calculate volume (uniform density = 1.0)
    volume = Utils.calculate_volume(grid)

    # 3. Create material model (standard linear elastic)
    λ, μ = create_material_model(E0, NU)

    # 4. Setup FEM problem
    print_info("Setting up FEM problem...")
    dh, cellvalues, K, f = setup_problem(grid)
    print_info("DOFs: $(ndofs(dh))")

    # 5. Assemble stiffness matrix
    print_info("Assembling stiffness matrix...")
    assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)

    # 6. Select boundary condition nodes
    print_info("Selecting boundary condition nodes...")
    symmetry_nodes = select_symmetry_nodes(grid)
    support_nodes = select_support_nodes(grid)
    force_nodes = select_semicircle_force_nodes(grid)

    print_info("Symmetry nodes (X-fixed): $(length(symmetry_nodes))")
    print_info("Support nodes (Y-fixed): $(length(support_nodes))")
    print_info("Force nodes (semicircle): $(length(force_nodes))")

    # 7. Apply boundary conditions
    print_info("Applying boundary conditions...")
    # Symmetry: fix X direction only (component 1)
    ch1 = apply_sliding_boundary!(K, f, dh, symmetry_nodes, [1])
    # Support: fix Y direction only (component 2)
    ch2 = apply_sliding_boundary!(K, f, dh, support_nodes, [2])

    # 8. Apply force
    print_info("Applying force: $FORCE_VECTOR")
    apply_force!(f, dh, collect(force_nodes), FORCE_VECTOR)

    # 9. Solve the system (direct solver)
    print_info("Solving linear system (direct solver)...")
    u, energy, stress_field, max_von_mises, max_stress_cell =
        solve_system(K, f, dh, cellvalues, λ, μ, ch1, ch2)

    # 10. Calculate compliance (C = f^T * u = 2 * U)
    compliance = dot(f, u)

    # 11. Print results
    print_success("\nRESULTS for $taskname:")
    print_data("  Compliance (f^T * u): $compliance")
    print_data("  Deformation energy: $energy J")
    print_data("  Volume: $volume")
    print_data("  Maximum von Mises stress: $max_von_mises at cell $max_stress_cell")

    # 12. Export results to TXT summary (in the same directory as input)
    txt_path = export_results_txt(taskname, compliance, volume, output_dir)
    print_info("Results summary saved to: $txt_path")

    return (
        name = taskname,
        compliance = compliance,
        energy = energy,
        volume = volume,
        max_von_mises = max_von_mises,
        max_stress_cell = max_stress_cell,
        n_elements = getncells(grid),
        n_nodes = getnnodes(grid),
        n_dofs = ndofs(dh),
    )
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

@testset "3D MBB Beam Compliance Evaluation" begin

    # Find all TET VTU files in the working directory
    print_info("Scanning directory: $WORK_DIR")
    input_files = find_tet_vtu_files(WORK_DIR)

    if isempty(input_files)
        print_warning("No *_TET.vtu files found in $WORK_DIR")
        @test false  # Fail if no files found
    else
        print_info("Found $(length(input_files)) TET mesh files to process")
        println()
    end

    # Storage for all results
    all_results = []

    # Process each input file
    for filepath in input_files
        # Generate task name from filename (without extension)
        taskname = splitext(basename(filepath))[1]

        @testset "$taskname" begin
            try
                # Output directory = same as input file location
                output_dir = dirname(filepath)
                result = analyze_tet_mesh(filepath, taskname, output_dir)
                push!(all_results, result)

                # Basic tests
                @test result.compliance > 0.0
                @test result.energy > 0.0
                @test result.volume > 0.0
            catch e
                print_error("Failed to analyze $taskname: $e")
                @test false
            end
        end
    end

    # ==========================================================================
    # SUMMARY COMPARISON
    # ==========================================================================

    if !isempty(all_results)
        print_info("\n" * "="^80)
        print_info("COMPLIANCE COMPARISON SUMMARY")
        print_info("="^80)

        # Print header
        println()
        println("┌" * "─"^50 * "┬" * "─"^15 * "┬" * "─"^15 * "┬" * "─"^12 * "┐")
        println(
            "│ " *
            rpad("Mesh Name", 48) *
            " │ " *
            lpad("Compliance", 13) *
            " │ " *
            lpad("Volume", 13) *
            " │ " *
            lpad("Elements", 10) *
            " │",
        )
        println("├" * "─"^50 * "┼" * "─"^15 * "┼" * "─"^15 * "┼" * "─"^12 * "┤")

        # Reference compliance (first result)
        ref_compliance = all_results[1].compliance

        for result in all_results
            name_short = length(result.name) > 48 ? result.name[1:45] * "..." : result.name

            println(
                "│ " *
                rpad(name_short, 48) *
                " │ " *
                lpad(@sprintf("%.6f", result.compliance), 13) *
                " │ " *
                lpad(@sprintf("%.6f", result.volume), 13) *
                " │ " *
                lpad(string(result.n_elements), 10) *
                " │",
            )
        end

        println("└" * "─"^50 * "┴" * "─"^15 * "┴" * "─"^15 * "┴" * "─"^12 * "┘")

        # Relative comparison
        println("\n" * "─"^95)
        println("RELATIVE COMPARISON (Reference: $(all_results[1].name))")
        println("─"^95)

        for (i, result) in enumerate(all_results)
            rel_diff = (result.compliance - ref_compliance) / ref_compliance * 100
            sign_str = rel_diff >= 0 ? "+" : ""
            println("  $(result.name):")
            println(
                "    Compliance relative difference: $(sign_str)$(round(rel_diff, digits=2))%",
            )
        end

        println("\n" * "="^95)
        print_success("Analysis completed for $(length(all_results)) meshes")
        println("\nResults saved to: $(WORK_DIR)/")
        println("="^95)
    end
end

# =============================================================================
# STANDALONE EXECUTION (when run directly, not as test)
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "="^80)
    println("3D MBB BEAM COMPLIANCE EVALUATION")
    println("="^80)
    println("\nGeometry: 2.0 × 1.0 × 1.0")
    println("Symmetry: Left face (x=0) - X direction fixed")
    println(
        "Support: Right bottom edge (x≥$(XMAX - SUPPORT_WIDTH), y=0) - Y direction fixed",
    )
    println("Load: Semicircle r=$(LOAD_RADIUS) at (0, $(YMAX), $(ZMAX/2)), F=$FORCE_VECTOR")
    println("Material: E=$(E0), ν=$(NU)")
    println("\nWorking directory: $(WORK_DIR)/")

    # List files that will be processed
    files = find_tet_vtu_files(WORK_DIR)
    println("\nTET mesh files to process ($(length(files))):")
    for (i, f) in enumerate(files)
        println("  $i. $(basename(f))")
    end

    println("="^80)
end
