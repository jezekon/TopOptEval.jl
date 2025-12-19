# =============================================================================
# 3D BEAM COMPLIANCE EVALUATION SCRIPT - GRIDAP VERSION
# =============================================================================
#
# Description:
#   Evaluates and compares compliance (deformation energy) for 5 different
#   mesh representations of the same 3D beam geometry with 4-corner fixation.
#
#   This version uses:
#   - TopOptEval.jl for SIMP meshes (variable density)
#   - Gridap.jl for solid tetrahedral meshes (verification)
#
# Input Files:
#   1. 1_3D_2x1x1_4Legs_SIMP.vtu          - Raw SIMP hexahedral mesh with density
#   2. 2_3D_2x1x1_4Legs_nodal_densities_TET.vtu - Tetrahedral mesh
#   3. 3_3D_2x1x1_4Legs_Smooth_SDF_TET.vtu      - Tetrahedral mesh (SDF smoothed)
#   4. 4_3D_2x1x1_4Legs-porous_TET.vtu          - Tetrahedral mesh (porous)
#   5. 5_3D_2x1x1_4Legs-custom_sdf_TET.vtu      - Tetrahedral mesh (custom SDF)
#
# Geometry:
#   - Domain: 2.0 × 1.0 × 1.0 (X × Y × Z)
#   - Fixed support: 4 corners at x=0 face (0.3 × 0.3 squares)
#   - Load: Circular region at x=2.0 face center, radius 0.1, force in -Z
#
# =============================================================================

using Test
using LinearAlgebra
using Printf

# TopOptEval for SIMP analysis
using TopOptEval
using TopOptEval.Utils
using Ferrite

# Gridap for solid mesh verification
using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.CellData
using Gridap.TensorValues

# ReadVTK for mesh import (shared by both approaches)
using ReadVTK

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input file paths
DATA_DIR = "data/3D_2x1x1_4Legs"
INPUT_FILES = [
    joinpath(DATA_DIR, "1_3D_2x1x1_4Legs_SIMP.vtu"),
    joinpath(DATA_DIR, "2_3D_2x1x1_4Legs_nodal_densities_TET.vtu"),
    joinpath(DATA_DIR, "3_3D_2x1x1_4Legs_Smooth_SDF_TET.vtu"),
    joinpath(DATA_DIR, "4_3D_2x1x1_4Legs-porous_TET.vtu"),
    joinpath(DATA_DIR, "5_3D_2x1x1_4Legs-custom_sdf_TET.vtu"),
]

# Mesh type indicators (true = SIMP with density, false = solid mesh)
IS_SIMP_MESH = [true, false, false, false, false]

# Geometry parameters
const XMAX = 2.0
const YMAX = 1.0
const ZMAX = 1.0
const FIX_SIZE = 0.3           # Size of corner fixation regions
const LOAD_RADIUS = 0.1        # Radius of circular load region

# Material properties
const E0 = 1.0                 # Young's modulus
const NU = 0.3                 # Poisson's ratio
const EMIN = 1e-9              # Minimum Young's modulus (for SIMP)
const P_SIMP = 3.0             # SIMP penalization power

# Lamé parameters (computed from E and ν)
const LAMBDA = E0 * NU / ((1 + NU) * (1 - 2 * NU))
const MU = E0 / (2 * (1 + NU))

# Force parameters
const FORCE_TOTAL = -1.0       # Total force in Z direction
const FORCE_VECTOR = [0.0, 0.0, FORCE_TOTAL]

# Tolerance for node/face selection
const NODE_SELECTION_TOL = 0.005

# Output directory for results
RESULTS_DIR = "results/3D_2x1x1_4Legs_gridap"

# Create results directory if it doesn't exist
mkpath(RESULTS_DIR)

# =============================================================================
# GRIDAP HELPER FUNCTIONS
# =============================================================================

"""
    vtu_to_gridap_model(filepath::String)

Converts a VTU file to a Gridap UnstructuredDiscreteModel.
Only supports tetrahedral meshes (VTK_TETRA = 10).

Parameters:
- filepath: Path to the VTU file

Returns:
- UnstructuredDiscreteModel ready for Gridap FEM analysis
"""
function vtu_to_gridap_model(filepath::String)
    println("  Converting VTU to Gridap model: $filepath")

    # Read VTU file (use qualified name to avoid conflict with Ferrite.VTKFile)
    vtk_file = ReadVTK.VTKFile(filepath)

    # Extract points (nodes) - returns 3×N matrix
    points_matrix = ReadVTK.get_points(vtk_file)
    n_nodes = size(points_matrix, 2)

    # Convert to vector of Points for Gridap
    node_coords = [
        Point(points_matrix[1, i], points_matrix[2, i], points_matrix[3, i]) for
        i = 1:n_nodes
    ]

    # Extract cells
    vtk_cells = ReadVTK.get_cells(vtk_file)
    connectivity = vtk_cells.connectivity
    offsets = vtk_cells.offsets
    types = vtk_cells.types

    # Create start indices for cell connectivity
    start_indices = vcat(1, offsets[1:(end-1)] .+ 1)

    # Filter only tetrahedral cells (VTK_TETRA = 10)
    tet_cells = Vector{Vector{Int}}()

    for i = 1:length(types)
        if types[i] == 10  # VTK_TETRA
            conn_indices = start_indices[i]:offsets[i]
            cell_conn = connectivity[conn_indices]
            push!(tet_cells, collect(cell_conn))
        end
    end

    if isempty(tet_cells)
        error("No tetrahedral cells found in VTU file")
    end

    println("    Found $(length(tet_cells)) tetrahedral cells, $n_nodes nodes")

    # Create Gridap Grid using the low-level constructor
    # Convert cell connectivity to the format Gridap expects
    cell_node_ids = Gridap.Arrays.Table(tet_cells)

    # Reference finite element for tetrahedra (linear, order 1)
    # Must use LagrangianRefFE, not just TET polytope
    reffe = LagrangianRefFE(Float64, TET, 1)
    reffes = [reffe]
    cell_types = fill(1, length(tet_cells))  # All cells are type 1 (TET)

    # Create the unstructured grid
    grid = UnstructuredGrid(node_coords, cell_node_ids, reffes, cell_types)

    # Create discrete model (wraps grid with topology info)
    model = UnstructuredDiscreteModel(grid)

    return model
end

"""
    is_on_fixed_corner(x)

Geometric predicate: returns true if point x is on one of the 4 fixed corners
at the x=0 face.

The 4 corners are:
- Bottom-left:  (y ≤ FIX_SIZE, z ≤ FIX_SIZE)
- Bottom-right: (y ≥ YMAX - FIX_SIZE, z ≤ FIX_SIZE)
- Top-left:     (y ≤ FIX_SIZE, z ≥ ZMAX - FIX_SIZE)
- Top-right:    (y ≥ YMAX - FIX_SIZE, z ≥ ZMAX - FIX_SIZE)
"""
function is_on_fixed_corner(x)
    tol = NODE_SELECTION_TOL

    # Must be on x=0 face
    if abs(x[1]) > tol
        return false
    end

    y, z = x[2], x[3]

    # Check each corner
    corner1 = (y <= FIX_SIZE + tol) && (z <= FIX_SIZE + tol)
    corner2 = (y >= YMAX - FIX_SIZE - tol) && (z <= FIX_SIZE + tol)
    corner3 = (y <= FIX_SIZE + tol) && (z >= ZMAX - FIX_SIZE - tol)
    corner4 = (y >= YMAX - FIX_SIZE - tol) && (z >= ZMAX - FIX_SIZE - tol)

    return corner1 || corner2 || corner3 || corner4
end

"""
    is_on_force_region(x)

Geometric predicate: returns true if point x is on the circular force region
at the center of x=XMAX face.
"""
function is_on_force_region(x)
    tol = NODE_SELECTION_TOL

    # Must be on x=XMAX face
    if abs(x[1] - XMAX) > tol
        return false
    end

    # Check if within circular region centered at (YMAX/2, ZMAX/2)
    y_center = YMAX / 2
    z_center = ZMAX / 2

    dist_sq = (x[2] - y_center)^2 + (x[3] - z_center)^2

    return dist_sq <= (LOAD_RADIUS + tol)^2
end

"""
    analyze_solid_mesh_gridap(filepath::String, taskname::String)

Analyzes a solid tetrahedral mesh using Gridap.jl for FEM computation.

This provides an independent verification of the TopOptEval.jl results.

Parameters:
- filepath: Path to the VTU file
- taskname: Name for output files

Returns:
- NamedTuple with analysis results
"""
function analyze_solid_mesh_gridap(filepath::String, taskname::String)
    print_info("\n" * "="^70)
    print_info("ANALYZING SOLID MESH WITH GRIDAP: $taskname")
    print_info("="^70)

    # =========================================================================
    # 1. Import mesh and create Gridap model
    # =========================================================================
    print_info("Importing mesh from: $filepath")
    model = vtu_to_gridap_model(filepath)

    # Get mesh statistics
    n_cells = num_cells(model)
    n_nodes = num_nodes(model)
    print_info("Mesh: $n_cells tetrahedra, $n_nodes nodes")

    # =========================================================================
    # 2. Define reference finite element and triangulations
    # =========================================================================
    print_info("Setting up FE spaces...")

    # Linear Lagrange elements for 3D elasticity (vector field)
    order = 1
    reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

    # Volume and boundary triangulations
    Ω = Triangulation(model)
    Γ = BoundaryTriangulation(model)  # All boundary faces

    # =========================================================================
    # 3. Create FE spaces with Dirichlet BC on fixed corners
    # =========================================================================
    # Test space with Dirichlet mask on fixed corners
    # The mask is a tuple of 3 functions (one per component) that return true
    # where that component should be constrained
    V = TestFESpace(
        model,
        reffe,
        conformity = :H1,
        dirichlet_masks = [(is_on_fixed_corner, is_on_fixed_corner, is_on_fixed_corner)],
    )

    # Trial space with zero displacement on fixed corners
    g = VectorValue(0.0, 0.0, 0.0)
    U = TrialFESpace(V, g)

    n_dofs = num_free_dofs(U)
    print_info("DOFs: $n_dofs (free)")

    # =========================================================================
    # 4. Define measures for integration
    # =========================================================================
    degree = 2 * order
    dΩ = Measure(Ω, degree)
    dΓ = Measure(Γ, degree)

    # =========================================================================
    # 5. Define constitutive law (linear elasticity)
    # =========================================================================
    # Stress tensor: σ = λ tr(ε) I + 2μ ε
    # Using local copies to avoid closure issues with global constants
    λ = LAMBDA
    μ = MU

    # Identity tensor for 3D symmetric tensors (must be defined as constant, not from CellField)
    I3 = one(SymTensorValue{3,Float64})

    # Constitutive relation using pre-defined identity tensor
    σ_elastic(ε) = λ * tr(ε) * I3 + 2 * μ * ε

    # =========================================================================
    # 6. Define weak form
    # =========================================================================
    print_info("Defining weak form...")

    # Bilinear form: a(u,v) = ∫ ε(v) : σ(ε(u)) dΩ
    a(u, v) = ∫(ε(v) ⊙ σ_elastic(ε(u))) * dΩ

    # Compute traction magnitude
    # Total force distributed over the circular region
    force_area = π * LOAD_RADIUS^2
    traction_magnitude = FORCE_TOTAL / force_area

    # Traction function: applies traction only in the force region
    function traction_field(x)
        if is_on_force_region(x)
            return VectorValue(0.0, 0.0, traction_magnitude)
        else
            return VectorValue(0.0, 0.0, 0.0)
        end
    end

    # Linear form: l(v) = ∫ v · t dΓ
    l(v) = ∫(v ⋅ traction_field) * dΓ

    # =========================================================================
    # 7. Assemble and solve
    # =========================================================================
    print_info("Assembling and solving system...")

    op = AffineFEOperator(a, l, U, V)
    uh = solve(op)

    # =========================================================================
    # 8. Post-processing: compute compliance and energy
    # =========================================================================
    print_info("Computing compliance and deformation energy...")

    # Deformation energy: U = 0.5 * ∫ ε(u) : σ(ε(u)) dΩ
    energy = 0.5 * sum(∫(ε(uh) ⊙ σ_elastic(ε(uh))) * dΩ)

    # For linear elasticity: Compliance C = f^T * u = 2 * U
    # Using this relation avoids numerical issues with traction field integration
    compliance = 2.0 * energy

    # Compute volume
    volume = sum(∫(1.0) * dΩ)

    # Von Mises stress computation is omitted for simplicity
    max_von_mises = 0.0
    max_stress_cell = 0

    # =========================================================================
    # 9. Print results
    # =========================================================================
    print_success("\nRESULTS for $taskname (GRIDAP):")
    print_data("  Compliance (2*U): $compliance")
    print_data("  Deformation energy: $energy J")
    print_data("  Volume: $volume")
    print_data("  Note: Max von Mises stress computation omitted")

    return (
        name = taskname,
        compliance = compliance,
        energy = energy,
        volume = volume,
        max_von_mises = max_von_mises,
        max_stress_cell = max_stress_cell,
        n_elements = n_cells,
        n_nodes = n_nodes,
        n_dofs = n_dofs,
        solver = "Gridap",
    )
end

# =============================================================================
# TOPOPTEVAL FUNCTIONS (for SIMP mesh - unchanged from original)
# =============================================================================

"""
    select_4corner_fixed_nodes(grid)

Selects nodes at the 4 corners of the x=0 face for fixed boundary conditions.
"""
function select_4corner_fixed_nodes(grid)
    fixed_nodes = Set{Int}()

    for node_id = 1:Ferrite.getnnodes(grid)
        coord = grid.nodes[node_id].x
        x, y, z = coord[1], coord[2], coord[3]

        if abs(x) < NODE_SELECTION_TOL
            in_corner1 =
                (y <= FIX_SIZE + NODE_SELECTION_TOL) && (z <= FIX_SIZE + NODE_SELECTION_TOL)
            in_corner2 =
                (y >= YMAX - FIX_SIZE - NODE_SELECTION_TOL) &&
                (z <= FIX_SIZE + NODE_SELECTION_TOL)
            in_corner3 =
                (y <= FIX_SIZE + NODE_SELECTION_TOL) &&
                (z >= ZMAX - FIX_SIZE - NODE_SELECTION_TOL)
            in_corner4 =
                (y >= YMAX - FIX_SIZE - NODE_SELECTION_TOL) &&
                (z >= ZMAX - FIX_SIZE - NODE_SELECTION_TOL)

            if in_corner1 || in_corner2 || in_corner3 || in_corner4
                push!(fixed_nodes, node_id)
            end
        end
    end

    return fixed_nodes
end

"""
    select_circular_force_nodes(grid)

Selects nodes in a circular region at the center of the x=xmax face.
"""
function select_circular_force_nodes(grid)
    force_nodes = Set{Int}()
    force_center_y = YMAX / 2
    force_center_z = ZMAX / 2

    for node_id = 1:Ferrite.getnnodes(grid)
        coord = grid.nodes[node_id].x
        x, y, z = coord[1], coord[2], coord[3]

        if abs(x - XMAX) < NODE_SELECTION_TOL
            dist_sq = (y - force_center_y)^2 + (z - force_center_z)^2
            if dist_sq <= (LOAD_RADIUS + NODE_SELECTION_TOL)^2
                push!(force_nodes, node_id)
            end
        end
    end

    if isempty(force_nodes)
        force_center = [XMAX, force_center_y, force_center_z]
        min_dist = Inf
        closest_node = 1
        for node_id = 1:Ferrite.getnnodes(grid)
            coord = grid.nodes[node_id].x
            dist = norm([coord[1], coord[2], coord[3]] - force_center)
            if dist < min_dist
                min_dist = dist
                closest_node = node_id
            end
        end
        push!(force_nodes, closest_node)
        @warn "No nodes found in circular region, using closest node: $closest_node"
    end

    return force_nodes
end

"""
    analyze_simp_mesh(filepath::String, taskname::String)

Analyzes a SIMP mesh with variable density field using TopOptEval.jl.
"""
function analyze_simp_mesh(filepath::String, taskname::String)
    print_info("\n" * "="^70)
    print_info("ANALYZING SIMP MESH WITH TOPOPTEVAL: $taskname")
    print_info("="^70)

    # Import mesh and density
    print_info("Importing mesh from: $filepath")
    grid = import_mesh(filepath)
    density_data = extract_cell_density(filepath)
    volume = Utils.calculate_volume(grid, density_data)

    # Create SIMP material model
    material_model = create_simp_material_model(E0, NU, EMIN, P_SIMP)

    # Setup FEM problem
    print_info("Setting up FEM problem...")
    dh, cellvalues, K, f = setup_problem(grid)
    print_info("DOFs: $(Ferrite.ndofs(dh))")

    # Assemble stiffness matrix with variable density
    print_info("Assembling stiffness matrix (SIMP)...")
    assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, density_data)

    # Select boundary condition nodes
    print_info("Selecting boundary condition nodes...")
    fixed_nodes = select_4corner_fixed_nodes(grid)
    force_nodes = select_circular_force_nodes(grid)
    print_info("Fixed corner nodes: $(length(fixed_nodes))")
    print_info("Force nodes (circular): $(length(force_nodes))")

    # Export boundary conditions
    export_boundary_conditions(
        grid,
        dh,
        fixed_nodes,
        force_nodes,
        joinpath(RESULTS_DIR, "$(taskname)_boundary_conditions"),
    )

    # Apply boundary conditions
    print_info("Applying boundary conditions...")
    ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)

    # Apply force
    print_info("Applying force: $FORCE_VECTOR")
    apply_force!(f, dh, collect(force_nodes), FORCE_VECTOR)

    # Solve
    print_info("Solving linear system...")
    u, energy, stress_field, max_von_mises, max_stress_cell =
        solve_system_simp(K, f, dh, cellvalues, material_model, density_data, ch1)

    # Calculate compliance
    compliance = dot(f, u)

    # Print results
    print_success("\nRESULTS for $taskname (TopOptEval):")
    print_data("  Compliance (f^T * u): $compliance")
    print_data("  Deformation energy: $energy J")
    print_data("  Volume (weighted): $volume")
    print_data("  Maximum von Mises stress: $max_von_mises at cell $max_stress_cell")

    # Export results
    export_results(u, dh, joinpath(RESULTS_DIR, "$(taskname)_displacement"))
    export_results(stress_field, dh, joinpath(RESULTS_DIR, "$(taskname)_stress"))

    return (
        name = taskname,
        compliance = compliance,
        energy = energy,
        volume = volume,
        max_von_mises = max_von_mises,
        max_stress_cell = max_stress_cell,
        n_elements = Ferrite.getncells(grid),
        n_nodes = Ferrite.getnnodes(grid),
        n_dofs = Ferrite.ndofs(dh),
        solver = "TopOptEval",
    )
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

@testset "3D Beam Compliance Evaluation (Gridap Verification)" begin

    all_results = []

    for (i, filepath) in enumerate(INPUT_FILES)
        if !isfile(filepath)
            print_warning("File not found: $filepath - skipping")
            continue
        end

        taskname = splitext(basename(filepath))[1]

        @testset "$taskname" begin
            try
                if IS_SIMP_MESH[i]
                    result = analyze_simp_mesh(filepath, taskname)
                else
                    result = analyze_solid_mesh_gridap(filepath, taskname)
                end

                push!(all_results, result)

                @test result.compliance > 0.0
                @test result.energy > 0.0
                @test result.volume > 0.0

            catch e
                print_error("Failed to analyze $taskname: $e")
                @test false
                rethrow(e)
            end
        end
    end

    # Summary table
    if !isempty(all_results)
        print_info("\n" * "="^90)
        print_info("COMPLIANCE COMPARISON SUMMARY")
        print_info("="^90)

        println()
        println(
            "┌" * "─"^35 * "┬" * "─"^12 * "┬" * "─"^14 * "┬" * "─"^14 * "┬" * "─"^10 * "┐",
        )
        println(
            "│ " *
            rpad("Mesh Name", 33) *
            " │ " *
            rpad("Solver", 10) *
            " │ " *
            lpad("Compliance", 12) *
            " │ " *
            lpad("Volume", 12) *
            " │ " *
            lpad("Elements", 8) *
            " │",
        )
        println(
            "├" * "─"^35 * "┼" * "─"^12 * "┼" * "─"^14 * "┼" * "─"^14 * "┼" * "─"^10 * "┤",
        )

        ref_compliance = all_results[1].compliance

        for result in all_results
            name_short = length(result.name) > 33 ? result.name[1:30] * "..." : result.name
            println(
                "│ " *
                rpad(name_short, 33) *
                " │ " *
                rpad(result.solver, 10) *
                " │ " *
                lpad(@sprintf("%.6f", result.compliance), 12) *
                " │ " *
                lpad(@sprintf("%.6f", result.volume), 12) *
                " │ " *
                lpad(string(result.n_elements), 8) *
                " │",
            )
        end

        println(
            "└" * "─"^35 * "┴" * "─"^12 * "┴" * "─"^14 * "┴" * "─"^14 * "┴" * "─"^10 * "┘",
        )

        # Relative comparison
        println("\n" * "─"^90)
        println("RELATIVE COMPARISON (Reference: $(all_results[1].name))")
        println("─"^90)

        for result in all_results
            rel_diff = (result.compliance - ref_compliance) / ref_compliance * 100
            sign_str = rel_diff >= 0 ? "+" : ""
            println("  [$(result.solver)] $(result.name):")
            println(
                "    Compliance relative difference: $(sign_str)$(round(rel_diff, digits=2))%",
            )
        end

        println("\n" * "="^90)
        print_success("Analysis completed for $(length(all_results)) meshes")
        println("="^90)
    end
end
