# =============================================================================
# 3D BEAM COMPLIANCE EVALUATION SCRIPT - GRIDAP VERSION (with BC Export)
# =============================================================================
#
# Description:
#   Evaluates and compares compliance (deformation energy) for 5 different
#   mesh representations of the same 3D beam geometry with 4-corner fixation.
#
#   This version uses:
#   - TopOptEval.jl for SIMP meshes (variable density)
#   - Gridap.jl for solid tetrahedral meshes (verification)
#   - Boundary conditions export for both solvers
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
using Gridap.Io  # For VTK export

# ReadVTK for mesh import (shared by both approaches)
using ReadVTK

# WriteVTK for manual VTU export (used for stress cell data and BC export)
using WriteVTK

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
    cell_node_ids = Gridap.Arrays.Table(tet_cells)

    # Reference finite element for tetrahedra (linear, order 1)
    reffe = LagrangianRefFE(Float64, TET, 1)
    reffes = [reffe]
    cell_types = fill(1, length(tet_cells))

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

# =============================================================================
# BOUNDARY CONDITIONS EXPORT FOR GRIDAP
# =============================================================================

"""
    export_gridap_boundary_conditions(model, output_path::String)

Exports boundary conditions for a Gridap model to VTU file for ParaView visualization.

Exports boundary faces (triangles for tet mesh) with cell data:
- boundary_type = 0: No boundary condition
- boundary_type = 1: Fixed (Dirichlet) boundary condition  
- boundary_type = 2: Force (Neumann) boundary condition

Parameters:
- model: Gridap UnstructuredDiscreteModel
- output_path: Output file path (without extension)

Returns:
- NamedTuple with statistics (n_fixed, n_force, n_faces)
"""
function export_gridap_boundary_conditions(model, output_path::String)
    print_info("Exporting Gridap boundary conditions to VTU...")

    # =========================================================================
    # 1. Get boundary triangulation from Gridap model
    # =========================================================================
    Γ = BoundaryTriangulation(model)

    # Get coordinates of each boundary face
    # Returns LazyArray of arrays, each containing vertex coordinates for one face
    cell_coords = get_cell_coordinates(Γ)
    n_faces = length(cell_coords)

    print_info("  Found $n_faces boundary faces")

    # =========================================================================
    # 2. Build boundary mesh for WriteVTK export
    # =========================================================================
    # Collect unique nodes from boundary faces and build connectivity
    node_list = Vector{NTuple{3,Float64}}()
    face_connectivity = Vector{Vector{Int}}()

    for face_coords in cell_coords
        face_node_ids = Int[]

        for coord in face_coords
            # Round coordinates for consistent hashing (avoid floating point issues)
            key = (
                round(coord[1], digits = 10),
                round(coord[2], digits = 10),
                round(coord[3], digits = 10),
            )

            # Find existing node or create new one
            node_id = findfirst(n -> n == key, node_list)
            if node_id === nothing
                push!(node_list, key)
                node_id = length(node_list)
            end
            push!(face_node_ids, node_id)
        end

        push!(face_connectivity, face_node_ids)
    end

    n_nodes = length(node_list)
    print_info("  Boundary mesh: $n_nodes nodes, $n_faces faces")

    # =========================================================================
    # 3. Compute BC type for each face based on centroid position
    # =========================================================================
    bc_types = zeros(Int, n_faces)

    for (i, face_coords) in enumerate(cell_coords)
        # Compute face centroid (average of vertex positions)
        centroid = sum(face_coords) / length(face_coords)

        # Evaluate geometric predicates
        if is_on_fixed_corner(centroid)
            bc_types[i] = 1  # Fixed BC
        elseif is_on_force_region(centroid)
            bc_types[i] = 2  # Force BC
        else
            bc_types[i] = 0  # No BC
        end
    end

    # Statistics
    n_fixed = count(==(1), bc_types)
    n_force = count(==(2), bc_types)

    print_info("  Fixed BC faces: $n_fixed")
    print_info("  Force BC faces: $n_force")

    # =========================================================================
    # 4. Convert to WriteVTK format and export
    # =========================================================================
    # Node coordinates as 3 × n_nodes matrix
    points = zeros(Float64, 3, n_nodes)
    for (i, node) in enumerate(node_list)
        points[1, i] = node[1]
        points[2, i] = node[2]
        points[3, i] = node[3]
    end

    # Create VTK cells (triangles for boundary of tet mesh)
    vtk_cells = WriteVTK.MeshCell[]
    for face_nodes in face_connectivity
        n_verts = length(face_nodes)
        if n_verts == 3
            push!(
                vtk_cells,
                WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TRIANGLE, face_nodes),
            )
        elseif n_verts == 4
            push!(vtk_cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUAD, face_nodes))
        else
            @warn "Unexpected boundary face with $n_verts vertices"
        end
    end

    # Write VTU file with BC markers as cell data
    vtk_grid(output_path, points, vtk_cells) do vtk
        vtk["boundary_type", WriteVTK.VTKCellData()] = bc_types
    end

    print_success("Boundary conditions exported to $(output_path).vtu")

    return (n_fixed = n_fixed, n_force = n_force, n_faces = n_faces)
end

# =============================================================================
# GRIDAP RESULTS EXPORT FUNCTIONS
# =============================================================================

"""
    compute_von_mises_stress(σ::SymTensorValue{3,Float64})

Computes von Mises equivalent stress from a 3D symmetric stress tensor.
"""
function compute_von_mises_stress(σ::SymTensorValue{3,Float64})
    σ_xx = σ[1, 1]
    σ_yy = σ[2, 2]
    σ_zz = σ[3, 3]
    σ_xy = σ[1, 2]
    σ_yz = σ[2, 3]
    σ_xz = σ[1, 3]

    vm = sqrt(
        σ_xx^2 + σ_yy^2 + σ_zz^2 - σ_xx * σ_yy - σ_yy * σ_zz - σ_zz * σ_xx +
        3 * (σ_xy^2 + σ_yz^2 + σ_xz^2),
    )

    return vm
end

"""
    export_gridap_results(Ω, uh, σ_field, output_path::String)

Exports Gridap FEM results to VTU file for visualization in ParaView.
Exports displacement as nodal data (fast).
"""
function export_gridap_results(Ω, uh, σ_field, output_path::String)
    print_info("Exporting Gridap displacement to VTU: $(output_path).vtu")
    writevtk(Ω, output_path, cellfields = ["displacement" => uh])
    print_success("Displacement exported to $(output_path).vtu")
end

"""
    export_gridap_full_results(model, Ω, dΩ, uh, σ_elastic, output_path::String)

Exports full Gridap FEM results including cell-averaged stress to VTU file.
Uses L2 projection to DG0 space for robust stress computation.
"""
function export_gridap_full_results(model, Ω, dΩ, uh, σ_elastic, output_path::String)
    print_info("Exporting Gridap results with stress to VTU...")

    # =========================================================================
    # 1. Extract mesh geometry for WriteVTK
    # =========================================================================
    grid = get_grid(model)
    n_cells = num_cells(model)
    n_nodes = num_nodes(model)

    print_info("  Mesh: $n_cells cells, $n_nodes nodes")

    # Node coordinates -> 3 x n_nodes matrix
    node_coords = get_node_coordinates(grid)
    points = zeros(Float64, 3, n_nodes)
    for (i, coord) in enumerate(node_coords)
        points[1, i] = coord[1]
        points[2, i] = coord[2]
        points[3, i] = coord[3]
    end

    # Cell connectivity -> VTK cells
    cell_node_ids = get_cell_node_ids(grid)
    vtk_cells = WriteVTK.MeshCell[]
    for cell_nodes in cell_node_ids
        push!(
            vtk_cells,
            WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TETRA, collect(cell_nodes)),
        )
    end

    # =========================================================================
    # 2. Extract nodal displacement
    # =========================================================================
    print_info("  Extracting nodal displacements...")

    disp_x = zeros(n_nodes)
    disp_y = zeros(n_nodes)
    disp_z = zeros(n_nodes)

    for (node_id, coord) in enumerate(node_coords)
        try
            u_val = uh(coord)
            disp_x[node_id] = u_val[1]
            disp_y[node_id] = u_val[2]
            disp_z[node_id] = u_val[3]
        catch e
            disp_x[node_id] = 0.0
            disp_y[node_id] = 0.0
            disp_z[node_id] = 0.0
        end
    end

    # =========================================================================
    # 3. Compute cell-averaged stress using L2 projection to DG0 space
    # =========================================================================
    print_info("  Computing cell-averaged stresses via L2 projection...")

    reffe_dg0 = ReferenceFE(lagrangian, Float64, 0)
    V_dg0 = TestFESpace(model, reffe_dg0, conformity = :L2)
    U_dg0 = TrialFESpace(V_dg0)

    σ = σ_elastic ∘ ε(uh)

    a_proj(u, v) = ∫(u * v) * dΩ

    e1 = VectorValue(1.0, 0.0, 0.0)
    e2 = VectorValue(0.0, 1.0, 0.0)
    e3 = VectorValue(0.0, 0.0, 1.0)

    function project_to_dg0(field_expr)
        l(v) = ∫(field_expr * v) * dΩ
        op = AffineFEOperator(a_proj, l, U_dg0, V_dg0)
        return solve(op)
    end

    print_info("    Projecting stress components...")
    σxx_h = project_to_dg0((e1 ⋅ σ) ⋅ e1)
    σyy_h = project_to_dg0((e2 ⋅ σ) ⋅ e2)
    σzz_h = project_to_dg0((e3 ⋅ σ) ⋅ e3)
    σxy_h = project_to_dg0((e1 ⋅ σ) ⋅ e2)
    σyz_h = project_to_dg0((e2 ⋅ σ) ⋅ e3)
    σxz_h = project_to_dg0((e1 ⋅ σ) ⋅ e3)

    cell_σ_xx = get_free_dof_values(σxx_h)
    cell_σ_yy = get_free_dof_values(σyy_h)
    cell_σ_zz = get_free_dof_values(σzz_h)
    cell_σ_xy = get_free_dof_values(σxy_h)
    cell_σ_yz = get_free_dof_values(σyz_h)
    cell_σ_xz = get_free_dof_values(σxz_h)

    # Compute von Mises from cell-averaged components
    cell_vm = zeros(n_cells)
    for i = 1:n_cells
        σxx, σyy, σzz = cell_σ_xx[i], cell_σ_yy[i], cell_σ_zz[i]
        σxy, σyz, σxz = cell_σ_xy[i], cell_σ_yz[i], cell_σ_xz[i]
        cell_vm[i] = sqrt(
            σxx^2 + σyy^2 + σzz^2 - σxx*σyy - σyy*σzz - σzz*σxx + 3*(σxy^2 + σyz^2 + σxz^2),
        )
    end

    max_vm = maximum(cell_vm)
    max_vm_cell = argmax(cell_vm)
    print_info("  Max von Mises stress: $max_vm at cell $max_vm_cell")

    # =========================================================================
    # 4. Write VTU file with WriteVTK
    # =========================================================================
    print_info("  Writing VTU file...")

    vtk_grid(output_path, points, vtk_cells) do vtk
        vtk["displacement", WriteVTK.VTKPointData()] = (disp_x, disp_y, disp_z)
        vtk["von_mises_stress", WriteVTK.VTKCellData()] = cell_vm
        vtk["stress_xx", WriteVTK.VTKCellData()] = collect(cell_σ_xx)
        vtk["stress_yy", WriteVTK.VTKCellData()] = collect(cell_σ_yy)
        vtk["stress_zz", WriteVTK.VTKCellData()] = collect(cell_σ_zz)
        vtk["stress_xy", WriteVTK.VTKCellData()] = collect(cell_σ_xy)
        vtk["stress_yz", WriteVTK.VTKCellData()] = collect(cell_σ_yz)
        vtk["stress_xz", WriteVTK.VTKCellData()] = collect(cell_σ_xz)
    end

    print_success("Full results exported to $(output_path).vtu")

    return max_vm, max_vm_cell
end

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

"""
    analyze_solid_mesh_gridap(filepath::String, taskname::String)

Analyzes a solid tetrahedral mesh using Gridap.jl for FEM computation.
Includes boundary condition export for visualization.

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

    n_cells = num_cells(model)
    n_nodes = num_nodes(model)
    print_info("Mesh: $n_cells tetrahedra, $n_nodes nodes")

    # =========================================================================
    # 2. Export boundary conditions for visualization
    # =========================================================================
    bc_output_path = joinpath(RESULTS_DIR, "$(taskname)_boundary_conditions")
    bc_stats = export_gridap_boundary_conditions(model, bc_output_path)

    # =========================================================================
    # 3. Define reference finite element and triangulations
    # =========================================================================
    print_info("Setting up FE spaces...")

    order = 1
    reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

    Ω = Triangulation(model)
    Γ = BoundaryTriangulation(model)

    # =========================================================================
    # 4. Create FE spaces with Dirichlet BC on fixed corners
    # =========================================================================
    V = TestFESpace(
        model,
        reffe,
        conformity = :H1,
        dirichlet_masks = [(is_on_fixed_corner, is_on_fixed_corner, is_on_fixed_corner)],
    )

    g = VectorValue(0.0, 0.0, 0.0)
    U = TrialFESpace(V, g)

    n_dofs = num_free_dofs(U)
    print_info("DOFs: $n_dofs (free)")

    # =========================================================================
    # 5. Define measures for integration
    # =========================================================================
    degree = 2 * order
    dΩ = Measure(Ω, degree)
    dΓ = Measure(Γ, degree)

    # =========================================================================
    # 6. Define constitutive law (linear elasticity)
    # =========================================================================
    λ = LAMBDA
    μ = MU
    I3 = one(SymTensorValue{3,Float64})
    σ_elastic(ε) = λ * tr(ε) * I3 + 2 * μ * ε

    # =========================================================================
    # 7. Define weak form
    # =========================================================================
    print_info("Defining weak form...")

    a(u, v) = ∫(ε(v) ⊙ σ_elastic(ε(u))) * dΩ

    force_area = π * LOAD_RADIUS^2
    traction_magnitude = FORCE_TOTAL / force_area

    function traction_field(x)
        if is_on_force_region(x)
            return VectorValue(0.0, 0.0, traction_magnitude)
        else
            return VectorValue(0.0, 0.0, 0.0)
        end
    end

    l(v) = ∫(v ⋅ traction_field) * dΓ

    # =========================================================================
    # 8. Assemble and solve
    # =========================================================================
    print_info("Assembling and solving system...")

    op = AffineFEOperator(a, l, U, V)
    uh = solve(op)

    # =========================================================================
    # 9. Post-processing: compute compliance and energy
    # =========================================================================
    print_info("Computing compliance and deformation energy...")

    energy = 0.5 * sum(∫(ε(uh) ⊙ σ_elastic(ε(uh))) * dΩ)
    compliance = 2.0 * energy
    volume = sum(∫(1.0) * dΩ)

    # =========================================================================
    # 10. Compute stress field and export results
    # =========================================================================
    print_info("Computing stress field...")
    σ_field = σ_elastic ∘ ε(uh)

    output_path = joinpath(RESULTS_DIR, "$(taskname)_gridap_results")
    max_von_mises, max_stress_cell =
        export_gridap_full_results(model, Ω, dΩ, uh, σ_elastic, output_path)

    # =========================================================================
    # 11. Print results
    # =========================================================================
    print_success("\nRESULTS for $taskname (GRIDAP):")
    print_data("  Compliance (2*U): $compliance")
    print_data("  Deformation energy: $energy J")
    print_data("  Volume: $volume")
    print_data("  Max von Mises stress: $max_von_mises at cell $max_stress_cell")
    print_data("  BC export: $(bc_stats.n_fixed) fixed, $(bc_stats.n_force) force faces")
    print_data("  Results exported to: $(output_path).vtu")
    print_data("  BC exported to: $(bc_output_path).vtu")

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
        bc_stats = bc_stats,
    )
end

# =============================================================================
# TOPOPTEVAL FUNCTIONS (for SIMP mesh)
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

    # Export boundary conditions (TopOptEval version)
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
