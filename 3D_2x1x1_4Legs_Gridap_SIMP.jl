# =============================================================================
# SIMP MESH ANALYSIS - GRIDAP (SIMPLE VERSION)
# =============================================================================
#
# Analyzes SIMP topology optimization results using Gridap.jl
# Inspired by GridapTopOpt's SmoothErsatzMaterialInterpolation approach.
#
# Key principle: Material properties are CellFields that vary with density,
# composed into the weak form via Gridap's field composition operator (∘).
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

# Paths
INPUT_FILE = "data/3D_2x1x1_4Legs/1_3D_2x1x1_4Legs_SIMP.vtu"
OUTPUT_DIR = "results/SIMP_gridap_standalone"

# Geometry
const XMAX = 2.0
const YMAX = 1.0
const ZMAX = 1.0
const FIX_SIZE = 0.3
const LOAD_RADIUS = 0.1

# Material
const E0 = 1.0
const NU = 0.3
const E_MIN = 1e-9
const P_SIMP = 3.0

# Lamé parameters for solid material
const LAMBDA = E0 * NU / ((1 + NU) * (1 - 2 * NU))
const MU = E0 / (2 * (1 + NU))

# Loading
const FORCE_TOTAL = -1.0

# Tolerance
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

function is_on_fixed_corner(coord)
    x, y, z = coord[1], coord[2], coord[3]
    abs(x) > TOL && return false

    c1 = (y <= FIX_SIZE + TOL) && (z <= FIX_SIZE + TOL)
    c2 = (y >= YMAX - FIX_SIZE - TOL) && (z <= FIX_SIZE + TOL)
    c3 = (y <= FIX_SIZE + TOL) && (z >= ZMAX - FIX_SIZE - TOL)
    c4 = (y >= YMAX - FIX_SIZE - TOL) && (z >= ZMAX - FIX_SIZE - TOL)

    return c1 || c2 || c3 || c4
end

function is_on_force_region(coord)
    x, y, z = coord[1], coord[2], coord[3]
    abs(x - XMAX) > TOL && return false

    y_c, z_c = YMAX / 2, ZMAX / 2
    dist_sq = (y - y_c)^2 + (z - z_c)^2
    return dist_sq <= (LOAD_RADIUS + TOL)^2
end

# =============================================================================
# SIMP MATERIAL INTERPOLATION (GridapTopOpt style)
# =============================================================================

"""
SIMP interpolation function: I(ρ) = E_min + (E_0 - E_min) * ρ^p

This is analogous to GridapTopOpt's ersatz interpolation:
    I(φ) = ε + (1-ε) * H_η(φ)

but for density-based (SIMP) rather than level-set based optimization.
"""
function simp_interp(ρ)
    return E_MIN + (E0 - E_MIN) * ρ^P_SIMP
end

# =============================================================================
# VTU IMPORT
# =============================================================================

# =============================================================================
# VTU IMPORT - FIXED VERSION
# =============================================================================
function import_simp_mesh(filepath::String)
    print_info("Importing mesh: $filepath")
    vtk = VTKFile(filepath)

    # Points
    pts = get_points(vtk)
    n_nodes = size(pts, 2)
    node_coords = [Point(pts[1, i], pts[2, i], pts[3, i]) for i = 1:n_nodes]
    print_info("  Nodes: $n_nodes")

    # Cells
    vtk_cells = get_cells(vtk)
    conn = vtk_cells.connectivity
    offs = vtk_cells.offsets
    types = vtk_cells.types
    n_cells = length(types)

    # Compute start indices for each cell
    start_indices = vcat(1, offs[1:(end-1)] .+ 1)

    # Extract hex cells (VTK type 12)
    # NOTE: ReadVTK returns 1-based connectivity in Julia - NO +1 needed!
    hex_cells = Vector{Vector{Int}}()
    hex_idx = Int[]

    for i = 1:n_cells
        if types[i] == 12  # VTK_HEXAHEDRON
            conn_indices = start_indices[i]:offs[i]
            cell_nodes = collect(conn[conn_indices])  # Already 1-based!
            push!(hex_cells, cell_nodes)
            push!(hex_idx, i)
        end
    end

    n_hex = length(hex_cells)
    print_info("  Hex cells: $n_hex")

    if n_hex == 0
        error("No hexahedral cells found!")
    end

    # Validate connectivity indices (helpful for debugging)
    all_indices = vcat(hex_cells...)
    min_idx, max_idx = extrema(all_indices)
    print_info("  Node index range in connectivity: [$min_idx, $max_idx]")

    if max_idx > n_nodes || min_idx < 1
        print_error("  Invalid connectivity: indices must be in [1, $n_nodes]")
        error("Mesh connectivity references non-existent nodes")
    end

    # Density field from cell data
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

    # Find fixed vertices
    fixed_vids = Int[]
    for (i, coord) in enumerate(node_coords)
        if is_on_fixed_corner(coord)
            push!(fixed_vids, i)
        end
    end
    print_info("  Fixed vertices: $(length(fixed_vids))")

    # Create Gridap model
    cell_node_ids = Gridap.Arrays.Table(hex_cells)
    reffe = LagrangianRefFE(Float64, HEX, 1)
    cell_types = fill(1, n_hex)

    grid = UnstructuredGrid(node_coords, cell_node_ids, [reffe], cell_types)
    model = UnstructuredDiscreteModel(grid)

    # Add "fixed" tag
    labels = get_face_labeling(model)
    v2e = get_face_entity(labels, 0)
    max_ent = maximum(v2e)
    fixed_ent = max_ent + 1

    for vid in fixed_vids
        v2e[vid] = fixed_ent
    end
    add_tag!(labels, "fixed", [fixed_ent])

    print_success("  Model created")
    return model, density_values, fixed_vids
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

    # 1. Import mesh
    model, ρ_values, fixed_vids = import_simp_mesh(INPUT_FILE)

    n_cells = num_cells(model)
    n_nodes = num_nodes(model)

    # 2. Triangulation and density CellField
    Ω = Triangulation(model)
    Γ = BoundaryTriangulation(model)

    # Create density as CellField (element-wise constant)
    ρ_h = CellField(ρ_values, Ω)

    # 3. SIMP material interpolation (GridapTopOpt style)
    # 
    # In GridapTopOpt they use:
    #   a(u,v,φ) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
    #
    # For SIMP we use:
    #   a(u,v) = ∫((I ∘ ρ_h)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
    #
    # where I(ρ) = E_min + (E_0 - E_min) * ρ^p

    # Effective modulus as CellField via composition
    I = simp_interp  # The interpolation function
    E_eff = I ∘ ρ_h  # Composed with density field

    # Lamé parameters as CellFields (proportional to E_eff)
    λ_h = (NU / ((1 + NU) * (1 - 2*NU))) * E_eff
    μ_h = (1 / (2 * (1 + NU))) * E_eff

    print_info("SIMP interpolation: E(ρ) = $E_MIN + ($E0 - $E_MIN) * ρ^$P_SIMP")

    # 4. FE Spaces
    order = 1
    reffe = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)

    V = TestFESpace(model, reffe, conformity = :H1, dirichlet_tags = ["fixed"])
    U = TrialFESpace(V, VectorValue(0.0, 0.0, 0.0))

    n_free = num_free_dofs(U)
    n_dir = num_dirichlet_dofs(U)

    print_info("DOFs: $n_free free, $n_dir Dirichlet")

    if n_dir == 0
        print_error("No Dirichlet DOFs - system singular!")
        error("BC application failed")
    end

    # 5. Measures
    degree = 2 * order
    dΩ = Measure(Ω, degree)
    dΓ = Measure(Γ, degree)

    # 6. Constitutive law with SIMP interpolation
    # 
    # σ(ε) = λ(ρ)*tr(ε)*I + 2*μ(ρ)*ε
    #
    # Using CellField arithmetic, this becomes spatially varying

    # I3 = one(SymTensorValue{3,Float64})
    # σ_simp(ε_val) = λ_h * tr(ε_val) * I3 + 2 * μ_h * ε_val

    # 7. Weak form
    # Bilinear form with density-dependent material
    # a(u, v) = ∫(ε(v) ⊙ (σ_simp ∘ ε(u))) * dΩ
    a(u, v) = ∫(λ_h * (tr(ε(u)) * tr(ε(v))) + 2 * μ_h * (ε(v) ⊙ ε(u))) * dΩ


    # Linear form (traction)
    force_area = π * LOAD_RADIUS^2
    traction_mag = FORCE_TOTAL / force_area

    function traction_field(x)
        if is_on_force_region(x)
            return VectorValue(0.0, 0.0, traction_mag)
        else
            return VectorValue(0.0, 0.0, 0.0)
        end
    end

    l(v) = ∫(v ⋅ traction_field) * dΓ

    # 8. Solve
    print_info("Solving...")

    op = AffineFEOperator(a, l, U, V)
    uh = solve(op)

    # 9. Post-processing
    u_vals = get_free_dof_values(uh)
    max_u = maximum(abs.(u_vals))
    print_info("Max displacement: $max_u")

    # Strain energy: U = 0.5 * ∫ ε : σ dΩ
    # energy = 0.5 * sum(∫(ε(uh) ⊙ (σ_simp ∘ ε(uh))) * dΩ)
    energy = 0.5 * sum(∫(λ_h * (tr(ε(uh)) * tr(ε(uh))) + 2 * μ_h * (ε(uh) ⊙ ε(uh))) * dΩ)

    # Compliance: C = 2U (for homogeneous Dirichlet BC)
    compliance = 2.0 * energy

    # Volume
    vol_total = sum(∫(1.0) * dΩ)
    vol_material = sum(∫(ρ_h) * dΩ)
    vol_frac = vol_material / vol_total

    # 10. Results
    println()
    println("="^70)
    println("RESULTS")
    println("="^70)
    print_success("Compliance:     $compliance")
    print_success("Strain energy:  $energy")
    print_success("Volume fraction: $(round(vol_frac*100, digits=2))%")
    print_success("Max displacement: $max_u")
    println()
    print_info("Mesh: $n_cells cells, $n_nodes nodes")
    print_info("Fixed vertices: $(length(fixed_vids)), Dirichlet DOFs: $n_dir")
    println()

    # 11. Export results
    export_path = joinpath(OUTPUT_DIR, "simp_results")

    # Simple VTU export using Gridap's writevtk
    writevtk(
        Ω,
        export_path,
        cellfields = ["displacement" => uh, "density" => ρ_h, "effective_E" => E_eff],
    )

    print_success("Results saved to: $export_path.vtu")

    return (
        compliance = compliance,
        energy = energy,
        vol_fraction = vol_frac,
        max_displacement = max_u,
    )
end

# =============================================================================
# RUN
# =============================================================================

# if abspath(PROGRAM_FILE) == @__FILE__
analyze_simp_gridap()
# end
