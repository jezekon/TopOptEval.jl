# =============================================================================
# SURFACE TRACTION IMPLEMENTATION
# =============================================================================
#
# Provides mesh-independent surface loading using proper Gauss quadrature
# integration over boundary facets. This approach ensures consistent results
# regardless of mesh refinement.
#
# NOTE: Uses get_face_nodes() defined in the parent FiniteElementAnalysis module
# to ensure consistent face definitions across all components.
#
# =============================================================================

export get_boundary_facets, apply_surface_traction!, compute_boundary_area

# Note: get_face_nodes is defined in parent module (FiniteElementAnalysis.jl)
# and is available here without explicit import

# =============================================================================
# BOUNDARY FACET IDENTIFICATION
# =============================================================================

"""
    get_boundary_facets(grid::Grid, nodes::Set{Int})

Identifies boundary facets (cell faces) where ALL vertices belong to the 
specified node set. This is used to define surfaces for traction application.

IMPORTANT: The returned local_face_id corresponds to Ferrite's face numbering,
which must match the get_face_nodes() definition for correct FacetValues integration.

# Arguments
- `grid::Grid`: Ferrite computational mesh
- `nodes::Set{Int}`: Set of node IDs defining the boundary region

# Returns
- `Set{Tuple{Int,Int}}`: Set of (cell_id, local_face_id) pairs

# Example
```julia
force_nodes = select_nodes_by_circle(grid, [2.0, 0.5, 0.5], [1.0, 0.0, 0.0], 0.1)
boundary_facets = get_boundary_facets(grid, force_nodes)
```
"""
function get_boundary_facets(grid::Grid, nodes::Set{Int})
    boundary_facets = Set{Tuple{Int,Int}}()

    for cell_id = 1:getncells(grid)
        cell = getcells(grid, cell_id)

        # Use shared face definition from parent module
        face_nodes_list = get_face_nodes(cell)

        for (local_face_id, face_nodes) in enumerate(face_nodes_list)
            # Get global node IDs for this face
            global_face_nodes = [cell.nodes[i] for i in face_nodes]

            # Check if ALL face nodes are in the specified node set
            if all(n -> n in nodes, global_face_nodes)
                push!(boundary_facets, (cell_id, local_face_id))
            end
        end
    end

    println("Found $(length(boundary_facets)) boundary facets")
    return boundary_facets
end

# =============================================================================
# BOUNDARY AREA COMPUTATION
# =============================================================================

"""
    compute_boundary_area(grid::Grid, dh::DofHandler, boundary_facets)

Computes the total area of the specified boundary facets using Gauss quadrature.
This is essential for converting total force to traction (t = F/A).

# Arguments
- `grid::Grid`: Ferrite computational mesh
- `dh::DofHandler`: Degree of freedom handler
- `boundary_facets`: Set of (cell_id, local_face_id) tuples

# Returns
- `Float64`: Total area of the boundary region

# Example
```julia
facets = get_boundary_facets(grid, force_nodes)
area = compute_boundary_area(grid, dh, facets)
traction = total_force / area
```
"""
function compute_boundary_area(grid::Grid, dh::DofHandler, boundary_facets)
    # Determine cell type and create appropriate FacetValues
    cell_type = typeof(getcells(grid, 1))

    if cell_type <: Ferrite.Hexahedron
        ip = Lagrange{RefHexahedron,1}()^3
        qr_face = FacetQuadratureRule{RefHexahedron}(2)
    elseif cell_type <: Ferrite.Tetrahedron
        ip = Lagrange{RefTetrahedron,1}()^3
        qr_face = FacetQuadratureRule{RefTetrahedron}(2)
    else
        error(
            "Unsupported cell type: $cell_type. Only Hexahedron and Tetrahedron supported.",
        )
    end

    facevalues = FacetValues(qr_face, ip)
    total_area = 0.0

    for (cell_id, local_face_id) in boundary_facets
        coords = getcoordinates(grid, cell_id)
        reinit!(facevalues, coords, local_face_id)

        # Integrate 1 over the face to get area
        for q_point = 1:getnquadpoints(facevalues)
            dΓ = getdetJdV(facevalues, q_point)
            total_area += dΓ
        end
    end

    return total_area
end

# =============================================================================
# SURFACE TRACTION APPLICATION
# =============================================================================

"""
    apply_surface_traction!(f, dh, grid, boundary_facets, traction_function)

Applies position-dependent surface traction using proper Gauss quadrature 
integration over boundary facets. This method is mesh-independent and 
physically accurate for distributed surface loads.

The weak form contribution is: ∫_Γ N · t dΓ

where N are shape functions and t is the traction vector.

# Arguments
- `f::Vector{Float64}`: Global force vector (modified in-place)
- `dh::DofHandler`: Degree of freedom handler
- `grid::Grid`: Ferrite computational mesh
- `boundary_facets`: Set of (cell_id, local_face_id) tuples from `get_boundary_facets`
- `traction_function::Function`: Function (x, y, z) -> [Tx, Ty, Tz] returning traction vector

# Example
```julia
# Uniform traction in -Z direction
force_total = -1.0
area = compute_boundary_area(grid, dh, facets)
traction_mag = force_total / area
traction_fn(x, y, z) = [0.0, 0.0, traction_mag]

apply_surface_traction!(f, dh, grid, facets, traction_fn)
```

# Notes
- For uniform loading, use `apply_uniform_surface_traction!` for convenience
- Total applied force equals ∫_Γ t dΓ (integrated traction over area)
- Results are mesh-independent unlike nodal force distribution
"""
function apply_surface_traction!(
    f::Vector{Float64},
    dh::DofHandler,
    grid::Grid,
    boundary_facets,
    traction_function::Function,
)
    # Determine cell type and create appropriate FacetValues
    cell_type = typeof(getcells(grid, 1))

    if cell_type <: Ferrite.Hexahedron
        ip = Lagrange{RefHexahedron,1}()^3
        qr_face = FacetQuadratureRule{RefHexahedron}(2)
    elseif cell_type <: Ferrite.Tetrahedron
        ip = Lagrange{RefTetrahedron,1}()^3
        qr_face = FacetQuadratureRule{RefTetrahedron}(2)
    else
        error(
            "Unsupported cell type: $cell_type. Only Hexahedron and Tetrahedron supported.",
        )
    end

    facevalues = FacetValues(qr_face, ip)

    n_basefuncs = getnbasefunctions(facevalues)
    fe = zeros(n_basefuncs)

    # Track total applied force for verification
    total_force = zeros(3)
    total_area = 0.0

    for (cell_id, local_face_id) in boundary_facets
        coords = getcoordinates(grid, cell_id)
        reinit!(facevalues, coords, local_face_id)

        fill!(fe, 0.0)

        for q_point = 1:getnquadpoints(facevalues)
            # Get integration weight (includes Jacobian determinant)
            dΓ = getdetJdV(facevalues, q_point)
            total_area += dΓ

            # Get spatial coordinates at quadrature point
            x_qp = spatial_coordinate(facevalues, q_point, coords)

            # Evaluate traction at this point
            traction = traction_function(x_qp[1], x_qp[2], x_qp[3])

            # Assemble element contribution: fe += N^T · t · dΓ
            for i = 1:n_basefuncs
                N = shape_value(facevalues, q_point, i)
                fe[i] += (N ⋅ traction) * dΓ
            end

            # Track total force
            total_force .+= traction .* dΓ
        end

        # Add element contributions to global force vector
        cell_dofs = celldofs(dh, cell_id)
        for (i, dof) in enumerate(cell_dofs)
            f[dof] += fe[i]
        end
    end

    println("Applied surface traction over $(length(boundary_facets)) facets")
    println("  Total boundary area: $(round(total_area, digits=6))")
    println(
        "  Total applied force: [$(round(total_force[1], digits=6)), $(round(total_force[2], digits=6)), $(round(total_force[3], digits=6))]",
    )
end

"""
    apply_uniform_surface_traction!(f, dh, grid, boundary_facets, total_force_vector)

Convenience function for applying uniform traction that results in a specified 
total force. Automatically computes the boundary area and distributes the force
as uniform traction t = F_total / A.

# Arguments
- `f::Vector{Float64}`: Global force vector (modified in-place)
- `dh::DofHandler`: Degree of freedom handler
- `grid::Grid`: Ferrite computational mesh
- `boundary_facets`: Set of (cell_id, local_face_id) tuples
- `total_force_vector::Vector{Float64}`: Desired total force [Fx, Fy, Fz]

# Example
```julia
facets = get_boundary_facets(grid, force_nodes)
apply_uniform_surface_traction!(f, dh, grid, facets, [0.0, 0.0, -1.0])
```

# Notes
- This is the recommended method for applying total force as surface load
- Ensures mesh-independent results unlike `apply_force!`
"""
function apply_uniform_surface_traction!(
    f::Vector{Float64},
    dh::DofHandler,
    grid::Grid,
    boundary_facets,
    total_force_vector::Vector{Float64},
)
    # Compute boundary area
    area = compute_boundary_area(grid, dh, boundary_facets)

    if area < 1e-12
        error("Boundary area is effectively zero. Check facet selection.")
    end

    # Compute uniform traction
    traction = total_force_vector ./ area

    println("Uniform surface traction:")
    println("  Boundary area: $(round(area, digits=6))")
    println("  Traction magnitude: $(round(norm(traction), digits=6))")

    # Create constant traction function
    traction_fn(x, y, z) = traction

    # Apply using general surface traction function
    apply_surface_traction!(f, dh, grid, boundary_facets, traction_fn)
end

