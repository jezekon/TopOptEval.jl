# =============================================================================
# 3D BEAM COMPLIANCE EVALUATION - GRIDAP VERSION
# =============================================================================
#
# Description:
#   Evaluates and compares compliance (deformation energy) for 3D beam geometry
#   with 4-corner fixation using both TopOptEval (SIMP) and Gridap (solid mesh).
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
using Gridap.Io
using Gridap.Arrays

using ReadVTK
using WriteVTK

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "data/3D_2x1x1_4Legs"
INPUT_FILES = [
    joinpath(DATA_DIR, "1_3D_2x1x1_4Legs_SIMP.vtu"),
    joinpath(DATA_DIR, "2_3D_2x1x1_4Legs_nodal_densities_TET.vtu"),
    joinpath(DATA_DIR, "3_3D_2x1x1_4Legs_Smooth_SDF_TET.vtu"),
    joinpath(DATA_DIR, "4_3D_2x1x1_4Legs-porous_TET.vtu"),
    joinpath(DATA_DIR, "5_3D_2x1x1_4Legs-custom_sdf_TET.vtu"),
]

IS_SIMP_MESH = [true, false, false, false, false]

# Geometry parameters
const XMAX = 2.0
const YMAX = 1.0
const ZMAX = 1.0
const FIX_SIZE = 0.3
const LOAD_RADIUS = 0.1

# Material properties
const E0 = 1.0
const NU = 0.3
const EMIN = 1e-9
const P_SIMP = 3.0

# Lamé parameters
const LAMBDA = E0 * NU / ((1 + NU) * (1 - 2 * NU))
const MU = E0 / (2 * (1 + NU))

# Force parameters
const FORCE_TOTAL = -1.0
const FORCE_VECTOR = [0.0, 0.0, FORCE_TOTAL]

# Tolerance for node selection
const NODE_SELECTION_TOL = 0.01

# Output directory
RESULTS_DIR = "results/3D_2x1x1_4Legs_gridap"
mkpath(RESULTS_DIR)

# =============================================================================
# OUTPUT HELPERS
# =============================================================================

print_info(msg) = printstyled("ℹ  $msg\n", color = :blue)
print_success(msg) = printstyled("✓  $msg\n", color = :green)
print_warning(msg) = printstyled("⚠  $msg\n", color = :yellow)
print_error(msg) = printstyled("✗  $msg\n", color = :red)

# =============================================================================
# GEOMETRIC PREDICATES
# =============================================================================

"""
Check if coordinate is on one of the 4 fixed corners at x=0 face.
"""
function is_on_fixed_corner(coord)
    tol = NODE_SELECTION_TOL
    x, y, z = coord[1], coord[2], coord[3]

    abs(x) > tol && return false

    corner1 = (y <= FIX_SIZE + tol) && (z <= FIX_SIZE + tol)
    corner2 = (y >= YMAX - FIX_SIZE - tol) && (z <= FIX_SIZE + tol)
    corner3 = (y <= FIX_SIZE + tol) && (z >= ZMAX - FIX_SIZE - tol)
    corner4 = (y >= YMAX - FIX_SIZE - tol) && (z >= ZMAX - FIX_SIZE - tol)

    return corner1 || corner2 || corner3 || corner4
end

"""
Check if coordinate is on the circular force region at x=XMAX face center.
"""
function is_on_force_region(coord)
    tol = NODE_SELECTION_TOL
    x, y, z = coord[1], coord[2], coord[3]

    abs(x - XMAX) > tol && return false

    y_center, z_center = YMAX / 2, ZMAX / 2
    dist_sq = (y - y_center)^2 + (z - z_center)^2

    return dist_sq <= (LOAD_RADIUS + tol)^2
end

"""
Check if a boundary face is in the force region.
Face must have ALL vertices on x=XMAX plane and centroid within circular region.
"""
function is_face_in_force_region(face_vertex_coords)
    tol = NODE_SELECTION_TOL

    # All vertices must be on x=XMAX plane
    for v in face_vertex_coords
        abs(v[1] - XMAX) > tol && return false
    end

    # Centroid must be in circular region
    centroid = sum(face_vertex_coords) / length(face_vertex_coords)
    y_center, z_center = YMAX / 2, ZMAX / 2
    dist_sq = (centroid[2] - y_center)^2 + (centroid[3] - z_center)^2

    return dist_sq <= (LOAD_RADIUS + tol)^2
end

# =============================================================================
# VTU TO GRIDAP MODEL CONVERSION
# =============================================================================

"""
Convert VTU file to Gridap model with proper vertex tags for BC application.
"""
function vtu_to_gridap_model(filepath::String)
    print_info("Converting VTU to Gridap model: $filepath")

    # Read VTU file
    vtk_file = ReadVTK.VTKFile(filepath)
    points_matrix = ReadVTK.get_points(vtk_file)
    n_nodes = size(points_matrix, 2)

    # Convert to Gridap Points
    node_coords = [
        Point(points_matrix[1, i], points_matrix[2, i], points_matrix[3, i]) for
        i = 1:n_nodes
    ]

    # Extract tetrahedral cells
    vtk_cells = ReadVTK.get_cells(vtk_file)
    connectivity = vtk_cells.connectivity
    offsets = vtk_cells.offsets
    types = vtk_cells.types
    start_indices = vcat(1, offsets[1:(end-1)] .+ 1)

    tet_cells = Vector{Vector{Int}}()
    for i = 1:length(types)
        if types[i] == 10  # VTK_TETRA
            conn_indices = start_indices[i]:offsets[i]
            push!(tet_cells, collect(connectivity[conn_indices]))
        end
    end

    isempty(tet_cells) && error("No tetrahedral cells found in VTU file")

    n_cells = length(tet_cells)
    print_info("  Nodes: $n_nodes, Tetrahedra: $n_cells")

    # Identify fixed and force vertices
    fixed_vertex_ids = Int[]
    force_vertex_ids = Int[]

    for (i, coord) in enumerate(node_coords)
        is_on_fixed_corner(coord) && push!(fixed_vertex_ids, i)
        is_on_force_region(coord) && push!(force_vertex_ids, i)
    end

    print_info("  Fixed vertices: $(length(fixed_vertex_ids))")
    print_info("  Force vertices: $(length(force_vertex_ids))")

    isempty(fixed_vertex_ids) && error("No fixed vertices found!")

    # Create Gridap grid
    cell_node_ids = Gridap.Arrays.Table(tet_cells)
    reffe = LagrangianRefFE(Float64, TET, 1)
    cell_types = fill(1, n_cells)
    grid = UnstructuredGrid(node_coords, cell_node_ids, [reffe], cell_types)

    # Create model and tag fixed vertices
    model = UnstructuredDiscreteModel(grid)
    labels = get_face_labeling(model)

    vertex_to_entity = get_face_entity(labels, 0)
    max_entity = maximum(vertex_to_entity)
    fixed_entity = max_entity + 1

    for vid in fixed_vertex_ids
        vertex_to_entity[vid] = fixed_entity
    end
    add_tag!(labels, "fixed", [fixed_entity])

    print_success("  Model created with boundary tags")

    return model, fixed_vertex_ids, force_vertex_ids
end

# =============================================================================
# BOUNDARY CONDITIONS EXPORT
# =============================================================================

"""
Export boundary conditions to VTU for ParaView visualization.
"""
function export_gridap_bc(model, fixed_nodes, force_nodes, output_path)
    Γ = BoundaryTriangulation(model)
    cell_coords = get_cell_coordinates(Γ)
    n_faces = length(cell_coords)

    # Build boundary mesh
    node_dict = Dict{NTuple{3,Float64},Int}()
    node_list = Vector{NTuple{3,Float64}}()
    face_connectivity = Vector{Vector{Int}}()

    for face_coords in cell_coords
        face_node_ids = Int[]
        for coord in face_coords
            key = (
                round(coord[1], digits = 10),
                round(coord[2], digits = 10),
                round(coord[3], digits = 10),
            )
            if haskey(node_dict, key)
                push!(face_node_ids, node_dict[key])
            else
                push!(node_list, key)
                node_dict[key] = length(node_list)
                push!(face_node_ids, length(node_list))
            end
        end
        push!(face_connectivity, face_node_ids)
    end

    # Compute BC type for each face
    bc_types = zeros(Int, n_faces)
    for (i, face_coords) in enumerate(cell_coords)
        centroid = sum(face_coords) / length(face_coords)
        if is_on_fixed_corner(centroid)
            bc_types[i] = 1
        elseif is_on_force_region(centroid)
            bc_types[i] = 2
        end
    end

    # Export to VTU
    points = zeros(Float64, 3, length(node_list))
    for (i, node) in enumerate(node_list)
        points[:, i] = [node[1], node[2], node[3]]
    end

    vtk_cells = WriteVTK.MeshCell[]
    for face_nodes in face_connectivity
        if length(face_nodes) == 3
            push!(
                vtk_cells,
                WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TRIANGLE, face_nodes),
            )
        elseif length(face_nodes) == 4
            push!(vtk_cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUAD, face_nodes))
        end
    end

    vtk_grid(output_path, points, vtk_cells) do vtk
        vtk["boundary_type", WriteVTK.VTKCellData()] = bc_types
    end

    n_fixed = count(==(1), bc_types)
    n_force = count(==(2), bc_types)
    print_info("  Boundary export: $n_fixed fixed, $n_force force faces")
end

# =============================================================================
# GRIDAP RESULTS EXPORT
# =============================================================================

"""
Export displacement and stress to VTU file.
"""
function export_gridap_results(model, Ω, dΩ, uh, σ_elastic, output_path)
    grid = get_grid(model)
    n_cells = num_cells(model)
    n_nodes = num_nodes(model)

    # Node coordinates
    node_coords = get_node_coordinates(grid)
    points = zeros(Float64, 3, n_nodes)
    for (i, coord) in enumerate(node_coords)
        points[:, i] = [coord[1], coord[2], coord[3]]
    end

    # Cell connectivity
    cell_node_ids = get_cell_node_ids(grid)
    vtk_cells = WriteVTK.MeshCell[]
    for cell_nodes in cell_node_ids
        push!(
            vtk_cells,
            WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TETRA, collect(cell_nodes)),
        )
    end

    # Extract nodal displacements
    disp_x, disp_y, disp_z = zeros(n_nodes), zeros(n_nodes), zeros(n_nodes)
    for (node_id, coord) in enumerate(node_coords)
        try
            u_val = uh(coord)
            disp_x[node_id], disp_y[node_id], disp_z[node_id] = u_val[1], u_val[2], u_val[3]
        catch
        end
    end

    # Compute cell-averaged stress via L2 projection to DG0
    reffe_dg0 = ReferenceFE(lagrangian, Float64, 0)
    V_dg0 = TestFESpace(model, reffe_dg0, conformity = :L2)
    U_dg0 = TrialFESpace(V_dg0)

    σ = σ_elastic ∘ ε(uh)
    a_proj(u, v) = ∫(u * v) * dΩ

    e1, e2, e3 =
        VectorValue(1.0, 0.0, 0.0), VectorValue(0.0, 1.0, 0.0), VectorValue(0.0, 0.0, 1.0)

    function project_to_dg0(field_expr)
        l(v) = ∫(field_expr * v) * dΩ
        op = AffineFEOperator(a_proj, l, U_dg0, V_dg0)
        return solve(op)
    end

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

    # Von Mises stress
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

    # Write VTU
    vtk_grid(output_path, points, vtk_cells) do vtk
        vtk["displacement", WriteVTK.VTKPointData()] = (disp_x, disp_y, disp_z)
        vtk["von_mises_stress", WriteVTK.VTKCellData()] = cell_vm
    end

    return max_vm, max_vm_cell
end

# =============================================================================
# GRIDAP SOLID MESH ANALYSIS
# =============================================================================

"""
Analyze a solid tetrahedral mesh using Gridap.
"""
function analyze_solid_mesh_gridap(filepath::String, taskname::String)
    println()
    println("="^70)
    print_info("ANALYZING SOLID MESH WITH GRIDAP: $taskname")
    println("="^70)

    # Import mesh with vertex tags
    model, fixed_vertex_ids, force_vertex_ids = vtu_to_gridap_model(filepath)

    n_cells = num_cells(model)
    n_nodes = num_nodes(model)

    # Export boundary conditions
    bc_output_path = joinpath(RESULTS_DIR, "$(taskname)_boundary_conditions")
    export_gridap_bc(model, fixed_vertex_ids, force_vertex_ids, bc_output_path)

    # Set up FE spaces
    order = 1
    reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

    Ω = Triangulation(model)
    Γ = BoundaryTriangulation(model)

    V = TestFESpace(model, reffe, conformity = :H1, dirichlet_tags = ["fixed"])
    U = TrialFESpace(V, VectorValue(0.0, 0.0, 0.0))

    n_free_dofs = num_free_dofs(U)
    n_dirichlet_dofs = num_dirichlet_dofs(U)

    print_info("DOFs: $n_free_dofs free, $n_dirichlet_dofs Dirichlet")

    n_dirichlet_dofs == 0 && error("No Dirichlet DOFs constrained!")

    # Measures and constitutive law
    degree = 2 * order
    dΩ = Measure(Ω, degree)
    dΓ = Measure(Γ, degree)

    λ, μ = LAMBDA, MU
    I3 = one(SymTensorValue{3,Float64})
    σ_elastic(ε) = λ * tr(ε) * I3 + 2 * μ * ε

    # Identify Neumann boundary faces
    face_coords_all = get_cell_coordinates(Γ)
    n_boundary_faces = num_cells(Γ)

    neumann_face_ids = Int[]
    for (i, face_coords) in enumerate(face_coords_all)
        is_face_in_force_region(face_coords) && push!(neumann_face_ids, i)
    end

    n_neumann = length(neumann_face_ids)
    print_info("Boundary faces: $n_boundary_faces total, $n_neumann in force region")

    n_neumann == 0 && error("No Neumann boundary faces found!")

    # Create characteristic function for Neumann faces
    χ_neumann = [i in neumann_face_ids ? 1.0 : 0.0 for i = 1:n_boundary_faces]
    χ_N = CellField(χ_neumann, Γ)

    # Compute Neumann boundary area
    neumann_area = sum(∫(χ_N) * dΓ)
    print_info("Neumann boundary area: $(round(neumann_area, digits=6))")

    neumann_area < 1e-12 && error("Neumann boundary area is zero!")

    # Compute traction
    traction_magnitude = FORCE_TOTAL / neumann_area
    print_info("Traction magnitude: $(round(traction_magnitude, digits=4))")

    t_N = VectorValue(0.0, 0.0, traction_magnitude)

    # Weak form
    a(u, v) = ∫(ε(v) ⊙ σ_elastic(ε(u))) * dΩ
    l(v) = ∫(χ_N * (v ⋅ t_N)) * dΓ

    # Solve
    print_info("Assembling and solving...")
    op = AffineFEOperator(a, l, U, V)
    uh = solve(op)

    # Post-processing
    u_free = get_free_dof_values(uh)
    max_disp = maximum(abs.(u_free))
    min_disp = minimum(u_free)

    print_info("Displacement range: [$min_disp, $max_disp]")

    energy = 0.5 * sum(∫(ε(uh) ⊙ σ_elastic(ε(uh))) * dΩ)
    compliance = 2.0 * energy
    volume = sum(∫(1.0) * dΩ)

    # Export results
    output_path = joinpath(RESULTS_DIR, "$(taskname)_results")
    max_von_mises, max_stress_cell =
        export_gridap_results(model, Ω, dΩ, uh, σ_elastic, output_path)

    # Print results
    println()
    println("="^50)
    print_success("RESULTS: $taskname")
    println("="^50)
    println("  Compliance:      $compliance")
    println("  Energy:          $energy J")
    println("  Volume:          $volume")
    println("  Max displacement: $max_disp")
    println("  Max von Mises:   $max_von_mises")

    return (
        name = taskname,
        compliance = compliance,
        energy = energy,
        volume = volume,
        max_von_mises = max_von_mises,
        max_stress_cell = max_stress_cell,
        n_elements = n_cells,
        n_nodes = n_nodes,
        n_dofs = n_free_dofs,
        n_dirichlet_dofs = n_dirichlet_dofs,
        max_displacement = max_disp,
        solver = "Gridap",
    )
end

# =============================================================================
# TOPOPTEVAL SIMP ANALYSIS
# =============================================================================

function select_4corner_fixed_nodes(grid)
    fixed_nodes = Set{Int}()
    for node_id = 1:Ferrite.getnnodes(grid)
        coord = grid.nodes[node_id].x
        x, y, z = coord[1], coord[2], coord[3]
        if abs(x) < NODE_SELECTION_TOL
            c1 =
                (y <= FIX_SIZE + NODE_SELECTION_TOL) && (z <= FIX_SIZE + NODE_SELECTION_TOL)
            c2 =
                (y >= YMAX - FIX_SIZE - NODE_SELECTION_TOL) &&
                (z <= FIX_SIZE + NODE_SELECTION_TOL)
            c3 =
                (y <= FIX_SIZE + NODE_SELECTION_TOL) &&
                (z >= ZMAX - FIX_SIZE - NODE_SELECTION_TOL)
            c4 =
                (y >= YMAX - FIX_SIZE - NODE_SELECTION_TOL) &&
                (z >= ZMAX - FIX_SIZE - NODE_SELECTION_TOL)
            (c1 || c2 || c3 || c4) && push!(fixed_nodes, node_id)
        end
    end
    return fixed_nodes
end

function select_circular_force_nodes(grid)
    force_nodes = Set{Int}()
    force_center_y, force_center_z = YMAX / 2, ZMAX / 2
    for node_id = 1:Ferrite.getnnodes(grid)
        coord = grid.nodes[node_id].x
        x, y, z = coord[1], coord[2], coord[3]
        if abs(x - XMAX) < NODE_SELECTION_TOL
            dist_sq = (y - force_center_y)^2 + (z - force_center_z)^2
            dist_sq <= (LOAD_RADIUS + NODE_SELECTION_TOL)^2 && push!(force_nodes, node_id)
        end
    end
    isempty(force_nodes) && @warn "No force nodes found!"
    return force_nodes
end

function analyze_simp_mesh(filepath::String, taskname::String)
    println()
    println("="^70)
    print_info("ANALYZING SIMP MESH WITH TOPOPTEVAL: $taskname")
    println("="^70)

    grid = import_mesh(filepath)
    density_data = extract_cell_density(filepath)
    volume = Utils.calculate_volume(grid, density_data)

    material_model = create_simp_material_model(E0, NU, EMIN, P_SIMP)

    dh, cellvalues, K, f = setup_problem(grid)
    print_info("DOFs: $(Ferrite.ndofs(dh))")

    assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, density_data)

    fixed_nodes = select_4corner_fixed_nodes(grid)
    force_nodes = select_circular_force_nodes(grid)
    print_info("Fixed nodes: $(length(fixed_nodes)), Force nodes: $(length(force_nodes))")

    export_boundary_conditions(
        grid,
        dh,
        fixed_nodes,
        force_nodes,
        joinpath(RESULTS_DIR, "$(taskname)_boundary_conditions"),
    )

    ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)
    apply_force!(f, dh, collect(force_nodes), FORCE_VECTOR)

    u, energy, stress_field, max_von_mises, max_stress_cell =
        solve_system_simp(K, f, dh, cellvalues, material_model, density_data, ch1)

    compliance = dot(f, u)

    println()
    println("="^50)
    print_success("RESULTS: $taskname")
    println("="^50)
    println("  Compliance:      $compliance")
    println("  Energy:          $energy J")
    println("  Volume:          $volume")
    println("  Max von Mises:   $max_von_mises")

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
# SUMMARY TABLE
# =============================================================================

function print_summary_table(all_results)
    println()
    println("="^100)
    println("COMPLIANCE COMPARISON SUMMARY")
    println("="^100)
    println()

    # Header
    println(
        "┌" *
        "─"^32 *
        "┬" *
        "─"^12 *
        "┬" *
        "─"^14 *
        "┬" *
        "─"^12 *
        "┬" *
        "─"^12 *
        "┬" *
        "─"^10 *
        "┐",
    )
    println(
        "│ " *
        rpad("Mesh Name", 30) *
        " │ " *
        rpad("Solver", 10) *
        " │ " *
        lpad("Compliance", 12) *
        " │ " *
        lpad("Max Disp", 10) *
        " │ " *
        lpad("Dir DOFs", 10) *
        " │ " *
        lpad("Elements", 8) *
        " │",
    )
    println(
        "├" *
        "─"^32 *
        "┼" *
        "─"^12 *
        "┼" *
        "─"^14 *
        "┼" *
        "─"^12 *
        "┼" *
        "─"^12 *
        "┼" *
        "─"^10 *
        "┤",
    )

    for result in all_results
        name_short = length(result.name) > 30 ? result.name[1:27] * "..." : result.name
        max_disp =
            haskey(result, :max_displacement) ? @sprintf("%.2e", result.max_displacement) :
            "N/A"
        dir_dofs =
            haskey(result, :n_dirichlet_dofs) ? string(result.n_dirichlet_dofs) : "N/A"

        println(
            "│ " *
            rpad(name_short, 30) *
            " │ " *
            rpad(result.solver, 10) *
            " │ " *
            lpad(@sprintf("%.6f", result.compliance), 12) *
            " │ " *
            lpad(max_disp, 10) *
            " │ " *
            lpad(dir_dofs, 10) *
            " │ " *
            lpad(string(result.n_elements), 8) *
            " │",
        )
    end

    println(
        "└" *
        "─"^32 *
        "┴" *
        "─"^12 *
        "┴" *
        "─"^14 *
        "┴" *
        "─"^12 *
        "┴" *
        "─"^12 *
        "┴" *
        "─"^10 *
        "┘",
    )

    # Relative comparison
    if length(all_results) > 1
        ref_compliance = all_results[1].compliance
        println()
        println("─"^100)
        println("RELATIVE COMPARISON (Reference: $(all_results[1].name))")
        println("─"^100)

        for result in all_results
            rel_diff = (result.compliance - ref_compliance) / ref_compliance * 100
            sign_str = rel_diff >= 0 ? "+" : ""
            println("  [$(result.solver)] $(result.name):")
            println("    Relative difference: $(sign_str)$(round(rel_diff, digits=2))%")
        end
    end

    print_success("\nAnalysis completed for $(length(all_results)) meshes")
    println("Results saved to: $RESULTS_DIR/")
end

# =============================================================================
# MAIN EXECUTION
# =============================================================================

@testset "3D Beam Compliance - Gridap" begin
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

                if haskey(result, :max_displacement)
                    @test result.max_displacement < 1e6
                    @test result.max_displacement > 0.0
                end

            catch e
                print_error("Failed to analyze $taskname: $e")
                @test false
                rethrow(e)
            end
        end
    end

    !isempty(all_results) && print_summary_table(all_results)
end
