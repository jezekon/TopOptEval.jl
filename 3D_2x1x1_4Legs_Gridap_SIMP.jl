# =============================================================================
# SIMP MESH ANALYSIS - GRIDAP (VERSION 3 - FIXED BC TAGGING)
# =============================================================================
#
# KEY FIXES:
# 1. Properly tag boundary FACES (not vertices) for Neumann BC
# 2. Use face vertex coordinates to identify force region (not just centroids)
# 3. Create separate BoundaryTriangulation for Neumann BC using face mask
# 4. Inspired by GridapTopOpt.jl boundary condition approach
#
# The previous version failed because:
# - Face centroids may not be exactly at x=XMAX due to mesh geometry
# - Traction function returning zero outside region results in zero force
# - Need to use GenericBoundaryTriangulation with face mask
#
# =============================================================================

using LinearAlgebra
using Statistics

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
OUTPUT_DIR = "results/SIMP_gridap_standalone"

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
# HELPER FUNCTIONS
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

    # Must be on the x=0 plane
    abs(x) > TOL && return false

    # Check if in any of the four corners
    c1 = (y <= FIX_SIZE + TOL) && (z <= FIX_SIZE + TOL)
    c2 = (y >= YMAX - FIX_SIZE - TOL) && (z <= FIX_SIZE + TOL)
    c3 = (y <= FIX_SIZE + TOL) && (z >= ZMAX - FIX_SIZE - TOL)
    c4 = (y >= YMAX - FIX_SIZE - TOL) && (z >= ZMAX - FIX_SIZE - TOL)

    return c1 || c2 || c3 || c4
end

"""
Check if a boundary face is in the force region.
A face is considered in the force region if:
1. ALL vertices are on the x=XMAX plane (within tolerance)
2. The face centroid is within the circular load region

This is more robust than just checking centroid coordinates.
"""
function is_face_in_force_region(face_vertex_coords)
    # First check: ALL vertices must be on x=XMAX plane
    for v in face_vertex_coords
        if abs(v[1] - XMAX) > TOL
            return false
        end
    end

    # Compute centroid for circle check
    centroid = sum(face_vertex_coords) / length(face_vertex_coords)

    # Check if centroid is in the circular region
    y_c, z_c = YMAX / 2, ZMAX / 2
    dist_sq = (centroid[2] - y_c)^2 + (centroid[3] - z_c)^2
    return dist_sq <= (LOAD_RADIUS + TOL)^2
end

"""
Check if a face is on the x=XMAX plane (for fallback).
"""
function is_face_on_xmax(face_vertex_coords)
    for v in face_vertex_coords
        if abs(v[1] - XMAX) > TOL
            return false
        end
    end
    return true
end

# =============================================================================
# SIMP MATERIAL INTERPOLATION
# =============================================================================

function simp_interp(ρ)
    return E_MIN + (E0 - E_MIN) * ρ^P_SIMP
end

# =============================================================================
# VTU IMPORT WITH PROPER BOUNDARY TAGGING
# =============================================================================

function import_simp_mesh(filepath::String)
    print_info("Importing mesh: $filepath")
    vtk = VTKFile(filepath)

    # =========================================================================
    # Extract points (nodes)
    # =========================================================================
    pts = get_points(vtk)
    n_nodes = size(pts, 2)
    node_coords = [Point(pts[1, i], pts[2, i], pts[3, i]) for i = 1:n_nodes]
    print_info("  Nodes: $n_nodes")

    # Debug: check node coordinate ranges
    x_coords = [c[1] for c in node_coords]
    y_coords = [c[2] for c in node_coords]
    z_coords = [c[3] for c in node_coords]
    print_info("  X range: [$(minimum(x_coords)), $(maximum(x_coords))]")
    print_info("  Y range: [$(minimum(y_coords)), $(maximum(y_coords))]")
    print_info("  Z range: [$(minimum(z_coords)), $(maximum(z_coords))]")

    # =========================================================================
    # Extract hexahedral cells
    # =========================================================================
    vtk_cells = get_cells(vtk)
    conn = vtk_cells.connectivity
    offs = vtk_cells.offsets
    types = vtk_cells.types
    n_cells = length(types)

    start_indices = vcat(1, offs[1:(end-1)] .+ 1)

    hex_cells = Vector{Vector{Int}}()
    hex_idx = Int[]

    # CRITICAL FIX: VTK hexahedron node ordering → Gridap lexicographic ordering
    # VTK HEX:    [0,1,2,3,4,5,6,7] = [(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)]
    # Gridap HEX: [0,1,2,3,4,5,6,7] = [(0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)]
    # Permutation (1-based): [1,2,4,3,5,6,8,7]
    vtk_to_gridap_hex = (1, 2, 4, 3, 5, 6, 8, 7)

    for i = 1:n_cells
        if types[i] == 12  # VTK_HEXAHEDRON
            conn_indices = start_indices[i]:offs[i]
            cell_nodes = collect(conn[conn_indices])

            # Validate we have 8 nodes
            if length(cell_nodes) != 8
                error("Cell $i is marked as HEX but has $(length(cell_nodes)) nodes")
            end

            # CRITICAL FIX: Convert 0-based → 1-based indexing if needed
            if minimum(cell_nodes) == 0
                cell_nodes .+= 1
            end

            # CRITICAL FIX: Reorder VTK node ordering → Gridap ordering
            cell_nodes = cell_nodes[collect(vtk_to_gridap_hex)]

            push!(hex_cells, cell_nodes)
            push!(hex_idx, i)
        end
    end

    n_hex = length(hex_cells)
    print_info("  Hex cells: $n_hex")

    if n_hex == 0
        error("No hexahedral cells found!")
    end

    # Validate connectivity
    all_indices = vcat(hex_cells...)
    min_idx, max_idx = extrema(all_indices)
    if max_idx > n_nodes || min_idx < 1
        error(
            "Mesh connectivity references non-existent nodes: min=$min_idx, max=$max_idx, n_nodes=$n_nodes",
        )
    end

    # =========================================================================
    # Extract density field from cell data
    # =========================================================================
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

    # =========================================================================
    # Create Gridap grid and model
    # =========================================================================
    cell_node_ids = Gridap.Arrays.Table(hex_cells)
    reffe = LagrangianRefFE(Float64, HEX, 1)
    cell_types = fill(1, n_hex)

    grid = UnstructuredGrid(node_coords, cell_node_ids, [reffe], cell_types)

    # Create model with automatic boundary detection
    model = UnstructuredDiscreteModel(grid)

    # =========================================================================
    # Identify fixed vertices (Dirichlet BC)
    # =========================================================================
    fixed_vids = Int[]
    for (i, coord) in enumerate(node_coords)
        if is_on_fixed_corner(coord)
            push!(fixed_vids, i)
        end
    end
    print_info("  Fixed vertices: $(length(fixed_vids))")

    # =========================================================================
    # Tag vertices for Dirichlet BC
    # =========================================================================
    labels = get_face_labeling(model)
    v2e = get_face_entity(labels, 0)  # 0-dim = vertices
    max_ent_v = maximum(v2e)
    fixed_ent = max_ent_v + 1

    for vid in fixed_vids
        v2e[vid] = fixed_ent
    end
    add_tag!(labels, "fixed", [fixed_ent])

    # =========================================================================
    # Verify boundary detection
    # =========================================================================
    Γ_test = BoundaryTriangulation(model)
    n_boundary_faces = num_cells(Γ_test)
    print_info("  Boundary faces detected: $n_boundary_faces")

    if n_boundary_faces == 0
        print_error("  No boundary faces detected! Model construction failed.")
        error("Boundary detection failed")
    end

    # =========================================================================
    # DIAGNOSTICS: Check boundary face coordinates
    # =========================================================================
    face_coords = get_cell_coordinates(Γ_test)

    # Check vertex X coordinates on boundary faces
    xmins = [minimum(v[1] for v in fc) for fc in face_coords]
    xmaxs = [maximum(v[1] for v in fc) for fc in face_coords]

    print_info("  Boundary vertex X range: [$(minimum(xmins)), $(maximum(xmaxs))]")

    faces_max_near_xmax = count(abs.(xmaxs .- XMAX) .<= TOL)
    faces_all_at_xmax =
        count((abs.(xmins .- XMAX) .<= TOL) .& (abs.(xmaxs .- XMAX) .<= TOL))

    print_info("  Faces with max X ≈ XMAX: $faces_max_near_xmax")
    print_info("  Faces with ALL vertices at X ≈ XMAX: $faces_all_at_xmax")

    if faces_all_at_xmax == 0
        print_warning("  No boundary faces found at x=XMAX! Check mesh orientation.")
    end

    print_success("  Model created with automatic boundary detection")

    return model, density_values, fixed_vids, node_coords
end

# =============================================================================
# CREATE NEUMANN BOUNDARY MASK
# =============================================================================

"""
Create a mask identifying which boundary faces are in the force region.
Returns a Vector{Bool} with length equal to number of boundary faces.
"""
function create_neumann_mask(model)
    Γ_all = BoundaryTriangulation(model)
    face_coords_all = get_cell_coordinates(Γ_all)

    n_boundary = num_cells(Γ_all)
    force_mask = Vector{Bool}(undef, n_boundary)

    force_count = 0
    xmax_count = 0

    for (i, face_coords) in enumerate(face_coords_all)
        # Check if face is in force region
        in_force = is_face_in_force_region(face_coords)
        force_mask[i] = in_force

        if in_force
            force_count += 1
        end

        # Also count faces on x=XMAX for debugging
        if is_face_on_xmax(face_coords)
            xmax_count += 1
        end
    end

    print_info("  Faces on x=XMAX plane: $xmax_count")
    print_info("  Faces in force region (circle): $force_count")

    if force_count == 0 && xmax_count > 0
        print_warning("  No faces in circular region, but $xmax_count on x=XMAX")
        print_warning("  Circle center: ($(YMAX/2), $(ZMAX/2)), radius: $LOAD_RADIUS")

        # Show some face centroids for debugging
        count_shown = 0
        for (i, face_coords) in enumerate(face_coords_all)
            if is_face_on_xmax(face_coords) && count_shown < 5
                centroid = sum(face_coords) / length(face_coords)
                y_c, z_c = YMAX / 2, ZMAX / 2
                dist = sqrt((centroid[2] - y_c)^2 + (centroid[3] - z_c)^2)
                print_info(
                    "    Face $i: centroid = $centroid, dist from center = $(round(dist, digits=4))",
                )
                count_shown += 1
            end
        end
    end

    return force_mask, xmax_count
end

# =============================================================================
# TAG NEUMANN BOUNDARY FACES
# =============================================================================

"""
Tag boundary faces in the force region for Neumann BC.
Returns vector of face IDs and count.
"""
function tag_neumann_boundary!(model)
    # Get boundary triangulation to iterate over faces
    Γ_all = BoundaryTriangulation(model)
    face_coords_all = get_cell_coordinates(Γ_all)

    n_boundary = num_cells(Γ_all)

    force_count = 0
    xmax_count = 0

    neumann_faces = Int[]

    for (local_id, face_coords) in enumerate(face_coords_all)
        # Check if face is in force region
        in_force = is_face_in_force_region(face_coords)

        if in_force
            push!(neumann_faces, local_id)
            force_count += 1
        end

        # Also count faces on x=XMAX for debugging
        if is_face_on_xmax(face_coords)
            xmax_count += 1
        end
    end

    print_info("  Faces on x=XMAX plane: $xmax_count")
    print_info("  Faces in force region (circle): $force_count")

    if force_count == 0 && xmax_count > 0
        print_warning("  No faces in circular region, but $xmax_count on x=XMAX")
        print_warning("  Using all x=XMAX faces as fallback")

        # Fallback: use all x=XMAX faces
        empty!(neumann_faces)
        for (local_id, face_coords) in enumerate(face_coords_all)
            if is_face_on_xmax(face_coords)
                push!(neumann_faces, local_id)
            end
        end
        return neumann_faces, xmax_count
    end

    return neumann_faces, force_count
end

# =============================================================================
# MAIN ANALYSIS (FIXED VERSION WITH FILTERED MEASURE)
# =============================================================================

function analyze_simp_gridap()
    println()
    println("="^70)
    println("SIMP MESH ANALYSIS - GRIDAP (V3 - FIXED BC TAGGING)")
    println("="^70)
    println()

    mkpath(OUTPUT_DIR)

    # =========================================================================
    # 1. Import mesh with density data
    # =========================================================================
    model, ρ_values, fixed_vids, node_coords = import_simp_mesh(INPUT_FILE)

    n_cells = num_cells(model)
    n_nodes = num_nodes(model)

    # =========================================================================
    # 2. Tag Neumann boundary faces
    # =========================================================================
    neumann_face_ids, n_neumann = tag_neumann_boundary!(model)

    if n_neumann == 0
        error("No suitable boundary found for Neumann BC!")
    end

    # =========================================================================
    # 3. Create triangulations
    # =========================================================================
    Ω = Triangulation(model)
    Γ_all = BoundaryTriangulation(model)
    n_boundary = num_cells(Γ_all)

    print_info("Boundary triangulation: $n_boundary faces total")
    print_info("Neumann boundary: $n_neumann faces")

    # =========================================================================
    # 4. Create characteristic function for Neumann faces
    # =========================================================================
    # χ_N = 1 on Neumann faces, 0 elsewhere
    χ_neumann = [i in neumann_face_ids ? 1.0 : 0.0 for i = 1:n_boundary]
    χ_N = CellField(χ_neumann, Γ_all)

    # =========================================================================
    # 5. Density CellField and SIMP interpolation
    # =========================================================================
    ρ_h = CellField(ρ_values, Ω)

    I = simp_interp
    E_eff = I ∘ ρ_h

    # Lamé parameters as CellFields
    λ_h = (NU / ((1 + NU) * (1 - 2 * NU))) * E_eff
    μ_h = (1 / (2 * (1 + NU))) * E_eff

    print_info("SIMP interpolation: E(ρ) = $E_MIN + ($E0 - $E_MIN) * ρ^$P_SIMP")

    # =========================================================================
    # 6. FE Spaces
    # =========================================================================
    order = 1
    reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

    V = TestFESpace(model, reffe, conformity = :H1, dirichlet_tags = ["fixed"])
    U = TrialFESpace(V, VectorValue(0.0, 0.0, 0.0))

    n_free = num_free_dofs(U)
    n_dir = num_dirichlet_dofs(U)

    print_info("DOFs: $n_free free, $n_dir Dirichlet")

    if n_dir == 0
        print_error("No Dirichlet DOFs - system will be singular!")
        error("BC application failed")
    end

    # =========================================================================
    # 7. Measures
    # =========================================================================
    degree = 2 * order
    dΩ = Measure(Ω, degree)
    dΓ_all = Measure(Γ_all, degree)

    # =========================================================================
    # 8. Compute Neumann boundary area using filtered measure
    # =========================================================================
    neumann_area = sum(∫(χ_N) * dΓ_all)
    print_info("Neumann boundary area: $(round(neumann_area, digits=6))")

    if neumann_area < 1e-12
        print_error("Neumann boundary area is zero! Check face tagging.")
        error("Invalid Neumann boundary")
    end

    # Calculate traction magnitude (uniform over Neumann boundary)
    traction_mag = FORCE_TOTAL / neumann_area
    print_info(
        "Traction magnitude: $(round(traction_mag, digits=4)) (uniform, in Z direction)",
    )

    # =========================================================================
    # 9. Weak form with filtered boundary measure
    # =========================================================================

    # Bilinear form with density-dependent material
    a(u, v) = ∫(λ_h * (tr(ε(u)) * tr(ε(v))) + 2 * μ_h * (ε(v) ⊙ ε(u))) * dΩ

    # Linear form - traction on Neumann boundary (filtered by χ_N)
    # Traction vector: (0, 0, traction_mag) - force in Z direction
    t_N = VectorValue(0.0, 0.0, traction_mag)

    # KEY: Multiply integrand by χ_N to restrict to Neumann faces
    l(v) = ∫(χ_N * (v ⋅ t_N)) * dΓ_all

    # =========================================================================
    # 10. Assemble and solve
    # =========================================================================
    print_info("Assembling and solving...")

    op = AffineFEOperator(a, l, U, V)
    uh = solve(op)

    # =========================================================================
    # 11. Post-processing
    # =========================================================================
    u_vals = get_free_dof_values(uh)
    max_u = maximum(abs.(u_vals))
    min_u = minimum(u_vals)

    print_info("Displacement range: [$min_u, $max_u]")

    if max_u < 1e-15
        print_error("Zero displacement detected!")
        print_error(
            "Check: 1) Boundary conditions, 2) Force application, 3) Material properties",
        )
    end

    # Strain energy
    energy = 0.5 * sum(∫(λ_h * (tr(ε(uh)) * tr(ε(uh))) + 2 * μ_h * (ε(uh) ⊙ ε(uh))) * dΩ)

    # Compliance (= 2 * strain energy = work done by external forces)
    compliance = 2.0 * energy

    # Also compute compliance as work: C = ∫ t · u dΓ (filtered)
    work = sum(∫(χ_N * (uh ⋅ t_N)) * dΓ_all)

    # Volume
    vol_total = sum(∫(1.0) * dΩ)
    vol_material = sum(∫(ρ_h) * dΩ)
    vol_frac = vol_material / vol_total

    # =========================================================================
    # 12. Print results
    # =========================================================================
    println()
    println("="^70)
    println("RESULTS")
    println("="^70)
    print_success("Compliance (2*energy):  $compliance")
    print_success("Compliance (work):      $work")
    print_success("Strain energy:          $energy")
    print_success("Volume fraction:        $(round(vol_frac*100, digits=2))%")
    print_success("Max displacement:       $max_u")
    println()
    print_info("Mesh: $n_cells cells, $n_nodes nodes")
    print_info("Fixed vertices: $(length(fixed_vids)), Dirichlet DOFs: $n_dir")
    print_info("Neumann boundary: $n_neumann faces, area: $(round(neumann_area, digits=6))")
    println()

    # =========================================================================
    # 13. Export results
    # =========================================================================
    export_path = joinpath(OUTPUT_DIR, "simp_results_v3")

    writevtk(
        Ω,
        export_path,
        cellfields = ["displacement" => uh, "density" => ρ_h, "effective_E" => E_eff],
    )

    print_success("Results saved to: $export_path.vtu")

    return (
        compliance = compliance,
        work = work,
        energy = energy,
        vol_fraction = vol_frac,
        max_displacement = max_u,
        neumann_area = neumann_area,
    )
end


# =============================================================================
# RUN
# =============================================================================

analyze_simp_gridap()
