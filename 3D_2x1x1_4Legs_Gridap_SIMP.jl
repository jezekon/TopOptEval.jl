# =============================================================================
# SIMP MESH ANALYSIS - GRIDAP
# =============================================================================
#
# Description:
#   Analyzes SIMP (Solid Isotropic Material with Penalization) hexahedral mesh
#   using Gridap for compliance evaluation.
#
# Geometry:
#   - Domain: 2.0 × 1.0 × 1.0 (X × Y × Z)
#   - Fixed support: 4 corners at x=0 face (0.3 × 0.3 squares)
#   - Load: Circular region at x=2.0 face center, radius 0.1, force in -Z
#
# =============================================================================

using LinearAlgebra
using Printf

using Gridap
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.CellData
using Gridap.TensorValues
using Gridap.Arrays

using ReadVTK
using ReadVTK: get_data
using WriteVTK

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = "data/3D_2x1x1_4Legs/1_3D_2x1x1_4Legs_SIMP.vtu"
OUTPUT_DIR = "results/SIMP_gridap"

# Geometry parameters
const XMAX = 2.0
const YMAX = 1.0
const ZMAX = 1.0
const FIX_SIZE = 0.3
const LOAD_RADIUS = 0.1

# Material properties
const E0 = 1.0
const NU = 0.3
const E_MIN = 1e-9
const P_SIMP = 3.0

# Loading
const FORCE_TOTAL = -1.0

# Tolerance for geometric comparisons
const TOL = 0.01

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
Check if a vertex coordinate is in the fixed corner region (Dirichlet BC).
Fixed corners are at x=0 plane, in the four corners of the YZ cross-section.
"""
function is_on_fixed_corner(coord)
    x, y, z = coord[1], coord[2], coord[3]

    abs(x) > TOL && return false

    c1 = (y <= FIX_SIZE + TOL) && (z <= FIX_SIZE + TOL)
    c2 = (y >= YMAX - FIX_SIZE - TOL) && (z <= FIX_SIZE + TOL)
    c3 = (y <= FIX_SIZE + TOL) && (z >= ZMAX - FIX_SIZE - TOL)
    c4 = (y >= YMAX - FIX_SIZE - TOL) && (z >= ZMAX - FIX_SIZE - TOL)

    return c1 || c2 || c3 || c4
end

"""
Check if a boundary face is in the force region.
Face must have ALL vertices on x=XMAX plane and centroid within circular region.
"""
function is_face_in_force_region(face_vertex_coords)
    # All vertices must be on x=XMAX plane
    for v in face_vertex_coords
        abs(v[1] - XMAX) > TOL && return false
    end

    # Centroid must be in circular region
    centroid = sum(face_vertex_coords) / length(face_vertex_coords)
    y_c, z_c = YMAX / 2, ZMAX / 2
    dist_sq = (centroid[2] - y_c)^2 + (centroid[3] - z_c)^2

    return dist_sq <= (LOAD_RADIUS + TOL)^2
end

"""
Check if a face is on the x=XMAX plane.
"""
function is_face_on_xmax(face_vertex_coords)
    for v in face_vertex_coords
        abs(v[1] - XMAX) > TOL && return false
    end
    return true
end

# =============================================================================
# SIMP MATERIAL INTERPOLATION
# =============================================================================

"""
SIMP interpolation: E(ρ) = E_MIN + (E0 - E_MIN) * ρ^p
"""
function simp_interp(ρ)
    return E_MIN + (E0 - E_MIN) * ρ^P_SIMP
end

# =============================================================================
# VTU IMPORT WITH BOUNDARY TAGGING
# =============================================================================

"""
Import SIMP mesh from VTU file with density field and proper boundary tags.
"""
function import_simp_mesh(filepath::String)
    print_info("Importing mesh: $filepath")
    vtk = VTKFile(filepath)

    # Extract points (nodes)
    pts = get_points(vtk)
    n_nodes = size(pts, 2)
    node_coords = [Point(pts[1, i], pts[2, i], pts[3, i]) for i = 1:n_nodes]
    print_info("  Nodes: $n_nodes")

    # Debug: coordinate ranges
    x_coords = [c[1] for c in node_coords]
    y_coords = [c[2] for c in node_coords]
    z_coords = [c[3] for c in node_coords]
    print_info("  X range: [$(minimum(x_coords)), $(maximum(x_coords))]")
    print_info("  Y range: [$(minimum(y_coords)), $(maximum(y_coords))]")
    print_info("  Z range: [$(minimum(z_coords)), $(maximum(z_coords))]")

    # Extract hexahedral cells
    vtk_cells = get_cells(vtk)
    conn = vtk_cells.connectivity
    offs = vtk_cells.offsets
    types = vtk_cells.types
    n_cells = length(types)

    start_indices = vcat(1, offs[1:(end-1)] .+ 1)

    hex_cells = Vector{Vector{Int}}()
    hex_idx = Int[]

    # VTK hexahedron node ordering → Gridap lexicographic ordering
    # VTK:    [(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)]
    # Gridap: [(0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)]
    vtk_to_gridap_hex = (1, 2, 4, 3, 5, 6, 8, 7)

    for i = 1:n_cells
        if types[i] == 12  # VTK_HEXAHEDRON
            conn_indices = start_indices[i]:offs[i]
            cell_nodes = collect(conn[conn_indices])

            length(cell_nodes) != 8 &&
                error("Cell $i is marked as HEX but has $(length(cell_nodes)) nodes")

            # Convert 0-based → 1-based if needed
            if minimum(cell_nodes) == 0
                @warn "Unexpected 0-based connectivity detected! Converting to 1-based."
                cell_nodes .+= 1
            elseif minimum(cell_nodes) < 1
                error("Invalid connectivity indices: minimum = $(minimum(cell_nodes))")
            end

            # Reorder VTK → Gridap
            cell_nodes = cell_nodes[collect(vtk_to_gridap_hex)]

            push!(hex_cells, cell_nodes)
            push!(hex_idx, i)
        end
    end

    n_hex = length(hex_cells)
    print_info("  Hex cells: $n_hex")

    isempty(hex_cells) && error("No hexahedral cells found!")

    # Validate connectivity
    all_indices = vcat(hex_cells...)
    min_idx, max_idx = extrema(all_indices)
    (max_idx > n_nodes || min_idx < 1) &&
        error("Invalid connectivity: min=$min_idx, max=$max_idx, n_nodes=$n_nodes")

    # Extract density field
    cell_data = get_cell_data(vtk)
    density_values = nothing

    for name in ["density", "Density", "rho", "material_density", "x"]
        try
            density_array = cell_data[name]
            all_dens = get_data(density_array)
            density_values = Float64[all_dens[i] for i in hex_idx]
            print_info("  Density field: '$name'")
            break
        catch
            continue
        end
    end

    if isnothing(density_values)
        print_warning("  No density field found, using uniform ρ=1.0")
        density_values = ones(n_hex)
    end

    print_info("  Density range: [$(minimum(density_values)), $(maximum(density_values))]")

    # Create Gridap grid and model
    cell_node_ids = Gridap.Arrays.Table(hex_cells)
    reffe = LagrangianRefFE(Float64, HEX, 1)
    cell_types = fill(1, n_hex)

    grid = UnstructuredGrid(node_coords, cell_node_ids, [reffe], cell_types)
    model = UnstructuredDiscreteModel(grid)

    # Identify fixed vertices
    fixed_vids = Int[]
    for (i, coord) in enumerate(node_coords)
        is_on_fixed_corner(coord) && push!(fixed_vids, i)
    end
    print_info("  Fixed vertices: $(length(fixed_vids))")

    # Tag vertices for Dirichlet BC
    labels = get_face_labeling(model)
    v2e = get_face_entity(labels, 0)
    max_ent_v = maximum(v2e)
    fixed_ent = max_ent_v + 1

    for vid in fixed_vids
        v2e[vid] = fixed_ent
    end
    add_tag!(labels, "fixed", [fixed_ent])

    # Verify boundary detection
    Γ_test = BoundaryTriangulation(model)
    n_boundary_faces = num_cells(Γ_test)
    print_info("  Boundary faces detected: $n_boundary_faces")

    n_boundary_faces == 0 && error("No boundary faces detected!")

    # Diagnostics: boundary face coordinates
    face_coords = get_cell_coordinates(Γ_test)
    xmins = [minimum(v[1] for v in fc) for fc in face_coords]
    xmaxs = [maximum(v[1] for v in fc) for fc in face_coords]

    print_info("  Boundary vertex X range: [$(minimum(xmins)), $(maximum(xmaxs))]")

    faces_max_near_xmax = count(abs.(xmaxs .- XMAX) .<= TOL)
    faces_all_at_xmax =
        count((abs.(xmins .- XMAX) .<= TOL) .& (abs.(xmaxs .- XMAX) .<= TOL))

    print_info("  Faces with max X ≈ XMAX: $faces_max_near_xmax")
    print_info("  Faces with ALL vertices at X ≈ XMAX: $faces_all_at_xmax")

    faces_all_at_xmax == 0 && print_warning("  No boundary faces found at x=XMAX!")

    print_success("  Model created with automatic boundary detection")

    return model, density_values, fixed_vids, node_coords
end

# =============================================================================
# TAG NEUMANN BOUNDARY FACES
# =============================================================================

"""
Identify boundary faces in the force region for Neumann BC.
"""
function tag_neumann_boundary(model)
    Γ_all = BoundaryTriangulation(model)
    face_coords_all = get_cell_coordinates(Γ_all)

    n_boundary = num_cells(Γ_all)
    force_count = 0
    xmax_count = 0

    neumann_faces = Int[]

    for (local_id, face_coords) in enumerate(face_coords_all)
        if is_face_in_force_region(face_coords)
            push!(neumann_faces, local_id)
            force_count += 1
        end

        is_face_on_xmax(face_coords) && (xmax_count += 1)
    end

    print_info("  Faces on x=XMAX plane: $xmax_count")
    print_info("  Faces in force region (circle): $force_count")

    if force_count == 0 && xmax_count > 0
        print_warning("  No faces in circular region, using all x=XMAX faces as fallback")
        empty!(neumann_faces)
        for (local_id, face_coords) in enumerate(face_coords_all)
            is_face_on_xmax(face_coords) && push!(neumann_faces, local_id)
        end
        return neumann_faces, xmax_count
    end

    return neumann_faces, force_count
end

# =============================================================================
# SUMMARY TABLE
# =============================================================================

function print_summary_table(result)
    println()
    println("="^80)
    println("COMPLIANCE SUMMARY")
    println("="^80)
    println()

    println("┌" * "─"^30 * "┬" * "─"^20 * "┐")
    println("│ " * rpad("Property", 28) * " │ " * lpad("Value", 18) * " │")
    println("├" * "─"^30 * "┼" * "─"^20 * "┤")
    println(
        "│ " *
        rpad("Compliance (2×energy)", 28) *
        " │ " *
        lpad(@sprintf("%.6f", result.compliance), 18) *
        " │",
    )
    println(
        "│ " *
        rpad("Compliance (work)", 28) *
        " │ " *
        lpad(@sprintf("%.6f", result.work), 18) *
        " │",
    )
    println(
        "│ " *
        rpad("Strain energy", 28) *
        " │ " *
        lpad(@sprintf("%.6f", result.energy), 18) *
        " │",
    )
    println(
        "│ " *
        rpad("Volume fraction", 28) *
        " │ " *
        lpad(@sprintf("%.2f%%", result.vol_fraction * 100), 18) *
        " │",
    )
    println(
        "│ " *
        rpad("Max displacement", 28) *
        " │ " *
        lpad(@sprintf("%.4f", result.max_displacement), 18) *
        " │",
    )
    println(
        "│ " *
        rpad("Neumann boundary area", 28) *
        " │ " *
        lpad(@sprintf("%.6f", result.neumann_area), 18) *
        " │",
    )
    println("├" * "─"^30 * "┼" * "─"^20 * "┤")
    println(
        "│ " * rpad("Elements", 28) * " │ " * lpad(string(result.n_elements), 18) * " │",
    )
    println("│ " * rpad("Nodes", 28) * " │ " * lpad(string(result.n_nodes), 18) * " │")
    println(
        "│ " * rpad("Free DOFs", 28) * " │ " * lpad(string(result.n_free_dofs), 18) * " │",
    )
    println(
        "│ " *
        rpad("Dirichlet DOFs", 28) *
        " │ " *
        lpad(string(result.n_dirichlet_dofs), 18) *
        " │",
    )
    println("└" * "─"^30 * "┴" * "─"^20 * "┘")

    println()
    print_success("Results saved to: $(result.output_path).vtu")
end

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

function analyze_simp_gridap()
    println()
    println("="^70)
    println("SIMP MESH ANALYSIS - GRIDAP")
    println("="^70)
    println()

    mkpath(OUTPUT_DIR)

    # Import mesh with density data
    model, ρ_values, fixed_vids, node_coords = import_simp_mesh(INPUT_FILE)

    n_cells = num_cells(model)
    n_nodes = num_nodes(model)

    # Tag Neumann boundary faces
    neumann_face_ids, n_neumann = tag_neumann_boundary(model)

    n_neumann == 0 && error("No suitable boundary found for Neumann BC!")

    # Create triangulations
    Ω = Triangulation(model)
    Γ_all = BoundaryTriangulation(model)
    n_boundary = num_cells(Γ_all)

    print_info("Boundary triangulation: $n_boundary faces total")
    print_info("Neumann boundary: $n_neumann faces")

    # Characteristic function for Neumann faces (χ_N = 1 on Neumann, 0 elsewhere)
    χ_neumann = [i in neumann_face_ids ? 1.0 : 0.0 for i = 1:n_boundary]
    χ_N = CellField(χ_neumann, Γ_all)

    # Density CellField and SIMP interpolation
    ρ_h = CellField(ρ_values, Ω)
    E_eff = simp_interp ∘ ρ_h

    # Lamé parameters as CellFields
    λ_h = (NU / ((1 + NU) * (1 - 2 * NU))) * E_eff
    μ_h = (1 / (2 * (1 + NU))) * E_eff

    print_info("SIMP interpolation: E(ρ) = $E_MIN + ($E0 - $E_MIN) * ρ^$P_SIMP")

    # FE Spaces
    order = 1
    reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

    V = TestFESpace(model, reffe, conformity = :H1, dirichlet_tags = ["fixed"])
    U = TrialFESpace(V, VectorValue(0.0, 0.0, 0.0))

    n_free = num_free_dofs(U)
    n_dir = num_dirichlet_dofs(U)

    print_info("DOFs: $n_free free, $n_dir Dirichlet")

    n_dir == 0 && error("No Dirichlet DOFs - system will be singular!")

    # Measures
    degree = 2 * order
    dΩ = Measure(Ω, degree)
    dΓ_all = Measure(Γ_all, degree)

    # Neumann boundary area
    neumann_area = sum(∫(χ_N) * dΓ_all)
    print_info("Neumann boundary area: $(round(neumann_area, digits=6))")

    neumann_area < 1e-12 && error("Neumann boundary area is zero!")

    # Traction magnitude
    traction_mag = FORCE_TOTAL / neumann_area
    print_info(
        "Traction magnitude: $(round(traction_mag, digits=4)) (uniform, in Z direction)",
    )

    # Weak form
    a(u, v) = ∫(λ_h * (tr(ε(u)) * tr(ε(v))) + 2 * μ_h * (ε(u) ⊙ ε(v))) * dΩ

    t_N = VectorValue(0.0, 0.0, traction_mag)
    l(v) = ∫(χ_N * (v ⋅ t_N)) * dΓ_all

    # Solve
    print_info("Assembling and solving...")
    op = AffineFEOperator(a, l, U, V)
    uh = solve(op)

    # Post-processing
    u_vals = get_free_dof_values(uh)
    max_u = maximum(abs.(u_vals))
    min_u = minimum(u_vals)

    print_info("Displacement range: [$min_u, $max_u]")

    max_u < 1e-15 && print_error("Zero displacement detected!")

    # Strain energy
    energy = 0.5 * sum(∫(λ_h * (tr(ε(uh)) * tr(ε(uh))) + 2 * μ_h * (ε(uh) ⊙ ε(uh))) * dΩ)

    # Compliance
    compliance = 2.0 * energy
    work = sum(∫(χ_N * (uh ⋅ t_N)) * dΓ_all)

    # Volume
    vol_total = sum(∫(1.0) * dΩ)
    vol_material = sum(∫(ρ_h) * dΩ)
    vol_frac = vol_material / vol_total

    # Export results
    export_path = joinpath(OUTPUT_DIR, "simp_results")
    writevtk(
        Ω,
        export_path,
        cellfields = ["displacement" => uh, "density" => ρ_h, "effective_E" => E_eff],
    )

    # Results structure
    result = (
        compliance = compliance,
        work = work,
        energy = energy,
        vol_fraction = vol_frac,
        max_displacement = max_u,
        neumann_area = neumann_area,
        n_elements = n_cells,
        n_nodes = n_nodes,
        n_free_dofs = n_free,
        n_dirichlet_dofs = n_dir,
        output_path = export_path,
    )

    # Print summary table
    print_summary_table(result)

    return result
end

# =============================================================================
# RUN
# =============================================================================

analyze_simp_gridap()
