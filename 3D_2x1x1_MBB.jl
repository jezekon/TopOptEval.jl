# =============================================================================
# 3D MBB BEAM COMPLIANCE EVALUATION SCRIPT
# =============================================================================
#
# Description:
#   Evaluates and compares compliance (deformation energy) for 5 different
#   mesh representations of the same 3D MBB beam geometry with symmetry BC.
#
# Input Files:
#   1. 1_3D_2x1x1_MBB_SIMP.vtu          - Raw SIMP hexahedral mesh with density
#   2. 2_3D_2x1x1_MBB_nodal_densities_TET.vtu - Tetrahedral mesh
#   3. 3_3D_2x1x1_MBB_Smooth_SDF_TET.vtu      - Tetrahedral mesh (SDF smoothed)
#   4. 4_3D_2x1x1_MBB-porous_TET.vtu          - Tetrahedral mesh (porous)
#   5. 5_3D_2x1x1_MBB-custom_sdf_TET.vtu      - Tetrahedral mesh (custom SDF)
#
# Geometry:
#   - Domain: 2.0 × 1.0 × 1.0 (X × Y × Z)
#   - Symmetry: Left face (x=0) - U1=0 only
#   - Support: Right bottom edge (x=2, y=0, all Z) - U2=0 only
#   - Load: Semicircle on top face at (0, 1, 0.5), radius 0.1, force in -Y
#
# =============================================================================

using Test
using LinearAlgebra
using Printf
using TopOptEval
using TopOptEval.Utils
using Ferrite  # For getnnodes, getncells, ndofs

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input file paths
DATA_DIR = "data/3D_2x1x1_MBB"
INPUT_FILES = [
    joinpath(DATA_DIR, "1_3D_2x1x1_MBB_SIMP.vtu"),
    joinpath(DATA_DIR, "2_3D_2x1x1_MBB_nodal_densities_TET.vtu"),
    # joinpath(DATA_DIR, "3_3D_2x1x1_MBB_Smooth_SDF_TET.vtu"),
    joinpath(DATA_DIR, "4_3D_2x1x1_MBB-porous_TET.vtu"),
    joinpath(DATA_DIR, "5_3D_2x1x1_MBB-custom_sdf_TET.vtu"),
]

# Mesh type indicators (true = SIMP with density, false = solid mesh)
IS_SIMP_MESH = [true, false, false, false, false]

# Geometry parameters
XMAX = 2.0
YMAX = 1.0
ZMAX = 1.0
SUPPORT_WIDTH = 0.05     # Width of support region at right edge (x >= XMAX - SUPPORT_WIDTH)
LOAD_RADIUS = 0.1        # Radius of semicircular load region

# Material properties
E0 = 1.0                 # Young's modulus
NU = 0.3                 # Poisson's ratio
EMIN = 1e-9              # Minimum Young's modulus (for SIMP)
P_SIMP = 3.0             # SIMP penalization power

# Force vector (MBB: force in -Y direction)
FORCE_VECTOR = [0.0, -1.0, 0.0]

# Tolerance for node selection (scaled for 2x1x1 domain)
NODE_SELECTION_TOL = 0.005

# Output directory for results
RESULTS_DIR = "results/3D_2x1x1_MBB"

# Create results directory if it doesn't exist
mkpath(RESULTS_DIR)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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
    analyze_simp_mesh(filepath::String, taskname::String)

Analyzes a SIMP mesh with variable density field.

Parameters:
- filepath: Path to the VTU file with density data
- taskname: Name for output files

Returns:
- NamedTuple with analysis results
"""
function analyze_simp_mesh(filepath::String, taskname::String)
    print_info("\n" * "="^70)
    print_info("ANALYZING SIMP MESH: $taskname")
    print_info("="^70)

    # 1. Import mesh
    print_info("Importing mesh from: $filepath")
    grid = import_mesh(filepath)

    # 2. Extract density data
    print_info("Extracting density data...")
    density_data = extract_cell_density(filepath)
    volume = Utils.calculate_volume(grid, density_data)

    # 3. Create SIMP material model
    material_model = create_simp_material_model(E0, NU, EMIN, P_SIMP)

    # 4. Setup FEM problem
    print_info("Setting up FEM problem...")
    dh, cellvalues, K, f = setup_problem(grid)
    print_info("DOFs: $(ndofs(dh))")

    # 5. Assemble stiffness matrix with variable density
    print_info("Assembling stiffness matrix (SIMP)...")
    assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, density_data)

    # 6. Select boundary condition nodes
    print_info("Selecting boundary condition nodes...")
    symmetry_nodes = select_symmetry_nodes(grid)
    support_nodes = select_support_nodes(grid)
    force_nodes = select_semicircle_force_nodes(grid)

    print_info("Symmetry nodes (X-fixed): $(length(symmetry_nodes))")
    print_info("Support nodes (Y-fixed): $(length(support_nodes))")
    print_info("Force nodes (semicircle): $(length(force_nodes))")

    # 7. Export boundary conditions for visualization
    all_constraint_nodes = union(symmetry_nodes, support_nodes)
    all_force_nodes = force_nodes
    export_boundary_conditions(
        grid,
        dh,
        all_constraint_nodes,
        all_force_nodes,
        joinpath(RESULTS_DIR, "$(taskname)_boundary_conditions"),
    )

    # 8. Apply boundary conditions
    print_info("Applying boundary conditions...")
    # Symmetry: fix X direction only (component 1)
    ch1 = apply_sliding_boundary!(K, f, dh, symmetry_nodes, [1])
    # Support: fix Y direction only (component 2)
    ch2 = apply_sliding_boundary!(K, f, dh, support_nodes, [2])

    # 9. Apply force
    print_info("Applying force: $FORCE_VECTOR")
    apply_force!(f, dh, collect(force_nodes), FORCE_VECTOR)

    # 10. Solve the system (direct solver)
    print_info("Solving linear system (direct solver)...")
    u, energy, stress_field, max_von_mises, max_stress_cell =
        solve_system_simp(K, f, dh, cellvalues, material_model, density_data, ch1, ch2)

    # 11. Calculate compliance (C = f^T * u = 2 * U)
    compliance = dot(f, u)

    # 12. Print results
    print_success("\nRESULTS for $taskname:")
    print_data("  Compliance (f^T * u): $compliance")
    print_data("  Deformation energy: $energy J")
    print_data("  Volume (weighted): $volume")
    print_data("  Maximum von Mises stress: $max_von_mises at cell $max_stress_cell")

    # 13. Export results
    export_results(u, dh, joinpath(RESULTS_DIR, "$(taskname)_displacement"))
    export_results(stress_field, dh, joinpath(RESULTS_DIR, "$(taskname)_stress"))

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

"""
    analyze_solid_mesh(filepath::String, taskname::String)

Analyzes a solid tetrahedral mesh without variable density.

Parameters:
- filepath: Path to the VTU file
- taskname: Name for output files

Returns:
- NamedTuple with analysis results
"""
function analyze_solid_mesh(filepath::String, taskname::String)
    print_info("\n" * "="^70)
    print_info("ANALYZING SOLID MESH: $taskname")
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

    # 7. Export boundary conditions for visualization
    all_constraint_nodes = union(symmetry_nodes, support_nodes)
    all_force_nodes = force_nodes
    export_boundary_conditions(
        grid,
        dh,
        all_constraint_nodes,
        all_force_nodes,
        joinpath(RESULTS_DIR, "$(taskname)_boundary_conditions"),
    )

    # 8. Apply boundary conditions
    print_info("Applying boundary conditions...")
    # Symmetry: fix X direction only (component 1)
    ch1 = apply_sliding_boundary!(K, f, dh, symmetry_nodes, [1])
    # Support: fix Y direction only (component 2)
    ch2 = apply_sliding_boundary!(K, f, dh, support_nodes, [2])

    # 9. Apply force
    print_info("Applying force: $FORCE_VECTOR")
    apply_force!(f, dh, collect(force_nodes), FORCE_VECTOR)

    # 10. Solve the system (direct solver)
    print_info("Solving linear system (direct solver)...")
    u, energy, stress_field, max_von_mises, max_stress_cell =
        solve_system(K, f, dh, cellvalues, λ, μ, ch1, ch2)

    # 11. Calculate compliance (C = f^T * u = 2 * U)
    compliance = dot(f, u)

    # 12. Print results
    print_success("\nRESULTS for $taskname:")
    print_data("  Compliance (f^T * u): $compliance")
    print_data("  Deformation energy: $energy J")
    print_data("  Volume: $volume")
    print_data("  Maximum von Mises stress: $max_von_mises at cell $max_stress_cell")

    # 13. Export results
    export_results(u, dh, joinpath(RESULTS_DIR, "$(taskname)_displacement"))
    export_results(stress_field, dh, joinpath(RESULTS_DIR, "$(taskname)_stress"))

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

    # Storage for all results
    all_results = []

    # Process each input file
    for (i, filepath) in enumerate(INPUT_FILES)
        # Skip if file doesn't exist
        if !isfile(filepath)
            print_warning("File not found: $filepath - skipping")
            continue
        end

        # Generate task name from filename
        taskname = splitext(basename(filepath))[1]

        @testset "$taskname" begin
            # Choose analysis method based on mesh type
            if IS_SIMP_MESH[i]
                result = analyze_simp_mesh(filepath, taskname)
            else
                result = analyze_solid_mesh(filepath, taskname)
            end

            push!(all_results, result)

            # Basic tests
            @test result.compliance > 0.0
            @test result.energy > 0.0
            @test result.volume > 0.0
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
        println("┌" * "─"^40 * "┬" * "─"^15 * "┬" * "─"^15 * "┬" * "─"^12 * "┐")
        println(
            "│ " *
            rpad("Mesh Name", 38) *
            " │ " *
            lpad("Compliance", 13) *
            " │ " *
            lpad("Volume", 13) *
            " │ " *
            lpad("Elements", 10) *
            " │",
        )
        println("├" * "─"^40 * "┼" * "─"^15 * "┼" * "─"^15 * "┼" * "─"^12 * "┤")

        # Reference compliance (first result, typically SIMP)
        ref_compliance = all_results[1].compliance

        for result in all_results
            rel_diff = (result.compliance - ref_compliance) / ref_compliance * 100
            name_short = length(result.name) > 38 ? result.name[1:35] * "..." : result.name

            println(
                "│ " *
                rpad(name_short, 38) *
                " │ " *
                lpad(@sprintf("%.6f", result.compliance), 13) *
                " │ " *
                lpad(@sprintf("%.6f", result.volume), 13) *
                " │ " *
                lpad(string(result.n_elements), 10) *
                " │",
            )
        end

        println("└" * "─"^40 * "┴" * "─"^15 * "┴" * "─"^15 * "┴" * "─"^12 * "┘")

        # Relative comparison
        println("\n" * "─"^80)
        println("RELATIVE COMPARISON (Reference: $(all_results[1].name))")
        println("─"^80)

        for (i, result) in enumerate(all_results)
            rel_diff = (result.compliance - ref_compliance) / ref_compliance * 100
            sign_str = rel_diff >= 0 ? "+" : ""
            println("  $(result.name):")
            println(
                "    Compliance relative difference: $(sign_str)$(round(rel_diff, digits=2))%",
            )
        end

        println("\n" * "="^80)
        print_success("Analysis completed for $(length(all_results)) meshes")
        println("\nResults saved to: $(RESULTS_DIR)/")
        println("="^80)
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
    println("\nInput files:")
    for (i, f) in enumerate(INPUT_FILES)
        exists = isfile(f) ? "✓" : "✗"
        mesh_type = IS_SIMP_MESH[i] ? "SIMP" : "Solid"
        println("  $exists $i. $f ($mesh_type)")
    end
    println("\nResults directory: $(RESULTS_DIR)/")
    println("="^80)
end
