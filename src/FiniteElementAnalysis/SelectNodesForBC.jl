"""
Surface Node Selection Module for Efficient Boundary Condition Application

This module provides optimized functions for selecting nodes on mesh surfaces
for boundary condition application. It pre-computes surface nodes to avoid
redundant calculations and ensures only boundary nodes are considered.
"""

using Ferrite
using LinearAlgebra

"""
    SurfaceNodeCache

Structure to cache pre-computed surface nodes and related data for efficient
boundary condition node selection.

Fields:
- `surface_nodes::Set{Int}`: Set of all nodes that lie on the mesh surface
- `node_coordinates::Vector{Vec{3,Float64}}`: Coordinates of surface nodes only
- `node_id_map::Dict{Int,Int}`: Maps global node IDs to indices in surface arrays
- `computed::Bool`: Flag indicating if cache has been computed
"""
mutable struct SurfaceNodeCache
    surface_nodes::Set{Int}
    node_coordinates::Vector{Vec{3,Float64}}
    node_id_map::Dict{Int,Int}
    computed::Bool

    # Constructor for empty cache
    SurfaceNodeCache() = new(Set{Int}(), Vec{3,Float64}[], Dict{Int,Int}(), false)
end

"""
    extract_surface_nodes!(cache::SurfaceNodeCache, grid::Grid)

Extracts all nodes that lie on the surface (boundary) of the mesh by identifying
faces that belong to only one element. Updates the provided cache structure.

Algorithm:
1. Generate all element faces with their connectivity
2. Count how many elements each face belongs to
3. Faces belonging to only one element are surface faces
4. Extract all unique nodes from surface faces

Parameters:
- `cache::SurfaceNodeCache`: Cache structure to populate (modified in-place)
- `grid::Grid`: Ferrite computational mesh

Returns:
- `nothing` (modifies cache in-place)
"""
function extract_surface_nodes!(cache::SurfaceNodeCache, grid::Grid)
    # Dictionary to store face connectivity and count occurrences
    # Key: sorted tuple of node IDs, Value: count of elements sharing this face
    face_count = Dict{Tuple{Vararg{Int}},Int}()

    # Iterate through all cells to find their faces
    for cell_idx = 1:getncells(grid)
        cell = getcells(grid, cell_idx)

        # Get all faces of the current cell
        faces = get_faces(cell)

        # Process each face
        for face_nodes in faces
            # Sort node IDs to create a canonical representation of the face
            # This ensures that faces shared by two elements are properly counted
            sorted_face = Tuple(sort(face_nodes))

            # Count how many elements this face belongs to
            face_count[sorted_face] = get(face_count, sorted_face, 0) + 1
        end
    end

    # Extract surface faces (faces that belong to only one element)
    surface_faces = [face for (face, count) in face_count if count == 1]

    # Extract unique surface nodes from all surface faces
    surface_node_set = Set{Int}()
    for face in surface_faces
        for node_id in face
            push!(surface_node_set, node_id)
        end
    end

    # Create mapping from global node ID to local index in surface arrays
    node_id_map = Dict{Int,Int}()
    node_coordinates = Vec{3,Float64}[]

    # Build coordinate array and mapping for surface nodes only
    for (local_idx, node_id) in enumerate(sort(collect(surface_node_set)))
        node_id_map[node_id] = local_idx

        # Get node coordinates and ensure 3D representation
        coord = grid.nodes[node_id].x
        if length(coord) == 3
            push!(node_coordinates, coord)
        else
            # Handle 2D case by padding with zero Z-coordinate
            push!(node_coordinates, Vec{3,Float64}(coord[1], coord[2], 0.0))
        end
    end

    # Update cache with computed data
    cache.surface_nodes = surface_node_set
    cache.node_coordinates = node_coordinates
    cache.node_id_map = node_id_map
    cache.computed = true

    println(
        "Surface extraction complete: $(length(surface_node_set)) surface nodes out of $(getnnodes(grid)) total nodes",
    )
    println(
        "Surface coverage: $(round(length(surface_node_set)/getnnodes(grid)*100, digits=1))%",
    )
end

"""
    get_faces(cell::Ferrite.AbstractCell)

Returns all faces of a given cell as vectors of node connectivity.
This is a generic dispatch function that handles different cell types.

Parameters:
- `cell`: Ferrite cell object

Returns:
- `Vector{Vector{Int}}`: Array of faces, each face is an array of node IDs
"""
function get_faces(cell::Ferrite.Tetrahedron)
    return [
        [cell.nodes[1], cell.nodes[2], cell.nodes[3]], # Face 1: nodes 1-2-3
        [cell.nodes[1], cell.nodes[2], cell.nodes[4]], # Face 2: nodes 1-2-4  
        [cell.nodes[2], cell.nodes[3], cell.nodes[4]], # Face 3: nodes 2-3-4
        [cell.nodes[1], cell.nodes[3], cell.nodes[4]],  # Face 4: nodes 1-3-4
    ]
end

function get_faces(cell::Ferrite.Hexahedron)
    return [
        [cell.nodes[1], cell.nodes[2], cell.nodes[3], cell.nodes[4]], # Bottom face (z=0)
        [cell.nodes[5], cell.nodes[6], cell.nodes[7], cell.nodes[8]], # Top face (z=1)
        [cell.nodes[1], cell.nodes[2], cell.nodes[6], cell.nodes[5]], # Front face (y=0)
        [cell.nodes[2], cell.nodes[3], cell.nodes[7], cell.nodes[6]], # Right face (x=1)
        [cell.nodes[3], cell.nodes[4], cell.nodes[8], cell.nodes[7]], # Back face (y=1)
        [cell.nodes[4], cell.nodes[1], cell.nodes[5], cell.nodes[8]],  # Left face (x=0)
    ]
end

function get_faces(cell::Ferrite.Triangle)
    return [
        [cell.nodes[1], cell.nodes[2]], # Edge 1: nodes 1-2
        [cell.nodes[2], cell.nodes[3]], # Edge 2: nodes 2-3
        [cell.nodes[3], cell.nodes[1]],  # Edge 3: nodes 3-1
    ]
end

function get_faces(cell::Ferrite.Quadrilateral)
    return [
        [cell.nodes[1], cell.nodes[2]], # Edge 1: nodes 1-2
        [cell.nodes[2], cell.nodes[3]], # Edge 2: nodes 2-3
        [cell.nodes[3], cell.nodes[4]], # Edge 3: nodes 3-4
        [cell.nodes[4], cell.nodes[1]],  # Edge 4: nodes 4-1
    ]
end

"""
    select_surface_nodes_by_plane(cache::SurfaceNodeCache,
                                  point::Vector{Float64}, 
                                  normal::Vector{Float64}, 
                                  tolerance::Float64=1.0)

Optimized version that selects surface nodes lying on a plane defined by point and normal.
Only considers nodes that are already identified as surface nodes.

Parameters:
- `cache::SurfaceNodeCache`: Pre-computed surface node data
- `point::Vector{Float64}`: A point on the plane [x, y, z]
- `normal::Vector{Float64}`: Normal vector to the plane [nx, ny, nz]
- `tolerance::Float64`: Distance tolerance for node selection

Returns:
- `Set{Int}`: Set of surface node IDs that lie on the specified plane

Throws:
- `ArgumentError`: If cache has not been computed yet
"""
function select_surface_nodes_by_plane(
    cache::SurfaceNodeCache,
    point::Vector{Float64},
    normal::Vector{Float64},
    tolerance::Float64 = 1.0,
)
    # Validate that cache has been computed
    if !cache.computed
        throw(
            ArgumentError(
                "Surface node cache has not been computed. Call extract_surface_nodes! first.",
            ),
        )
    end

    # Normalize the normal vector to ensure consistent distance calculations
    unit_normal = normal / norm(normal)

    # Initialize set to store selected nodes
    selected_nodes = Set{Int}()

    # Iterate only through surface nodes (much faster than all nodes)
    for node_id in cache.surface_nodes
        # Get pre-computed coordinates from cache
        local_idx = cache.node_id_map[node_id]
        coord = cache.node_coordinates[local_idx]

        # Calculate perpendicular distance from node to plane
        # Distance formula: |((node - point) · normal)| / |normal|
        # Since normal is already normalized, we can omit division
        dist = abs(dot(coord - point, unit_normal))

        # Select node if it's within specified tolerance of the plane
        if dist < tolerance
            push!(selected_nodes, node_id)
        end
    end

    println("Selected $(length(selected_nodes)) surface nodes on the specified plane")
    println("Searched through $(length(cache.surface_nodes)) surface nodes")

    return selected_nodes
end

"""
    select_surface_nodes_by_circle(cache::SurfaceNodeCache,
                                   center::Vector{Float64}, 
                                   normal::Vector{Float64}, 
                                   radius::Float64, 
                                   tolerance::Float64=1.0)

Optimized version that selects surface nodes within a circular region defined by 
center, normal vector, and radius. Only considers surface nodes.

Parameters:
- `cache::SurfaceNodeCache`: Pre-computed surface node data  
- `center::Vector{Float64}`: Center of the circle [x, y, z]
- `normal::Vector{Float64}`: Normal vector to the plane containing the circle [nx, ny, nz]
- `radius::Float64`: Radius of the circle
- `tolerance::Float64`: Distance tolerance for node selection

Returns:
- `Set{Int}`: Set of surface node IDs that lie within the circular region

Throws:
- `ArgumentError`: If cache has not been computed yet
"""
function select_surface_nodes_by_circle(
    cache::SurfaceNodeCache,
    center::Vector{Float64},
    normal::Vector{Float64},
    radius::Float64,
    tolerance::Float64 = 1.0,
)
    # Validate that cache has been computed
    if !cache.computed
        throw(
            ArgumentError(
                "Surface node cache has not been computed. Call extract_surface_nodes! first.",
            ),
        )
    end

    # First, find surface nodes that lie on the plane containing the circle
    nodes_on_plane = select_surface_nodes_by_plane(cache, center, normal, tolerance)

    # Normalize the normal vector for consistent calculations
    unit_normal = normal / norm(normal)

    # Initialize set for nodes within the circular region
    nodes_in_circle = Set{Int}()

    # Check which plane nodes are within the circle radius
    for node_id in nodes_on_plane
        # Get node coordinates from cache
        local_idx = cache.node_id_map[node_id]
        coord = cache.node_coordinates[local_idx]

        # Calculate vector from circle center to node
        center_to_node = coord - center

        # Project this vector onto the plane (remove component along normal)
        # This gives us the position of the node within the plane
        plane_projection = center_to_node - dot(center_to_node, unit_normal) * unit_normal

        # Calculate distance from center within the plane
        planar_distance = norm(plane_projection)

        # Include node if it's within the circle radius (with tolerance)
        if planar_distance <= radius + tolerance
            push!(nodes_in_circle, node_id)
        end
    end

    println("Selected $(length(nodes_in_circle)) surface nodes in the circular region")

    return nodes_in_circle
end

# Convenience functions with automatic caching for backward compatibility

# Global cache storage for automatic caching (one cache per grid)
const GRID_CACHE_STORAGE = Dict{UInt,SurfaceNodeCache}()

"""
    get_or_create_cache(grid::Grid)

Internal function to get or create a surface node cache for a given grid.
Uses grid object hash as key for caching.

Parameters:
- `grid::Grid`: Ferrite mesh

Returns:
- `SurfaceNodeCache`: Cache object (computed if necessary)
"""
function get_or_create_cache(grid::Grid)
    # Use grid object hash as unique identifier
    grid_hash = hash(grid)

    # Return existing cache or create new one
    if haskey(GRID_CACHE_STORAGE, grid_hash)
        cache = GRID_CACHE_STORAGE[grid_hash]
        if !cache.computed
            extract_surface_nodes!(cache, grid)
        end
        return cache
    else
        # Create new cache and compute surface nodes
        cache = SurfaceNodeCache()
        extract_surface_nodes!(cache, grid)
        GRID_CACHE_STORAGE[grid_hash] = cache
        return cache
    end
end

"""
    select_nodes_by_plane(grid::Grid, 
                          point::Vector{Float64}, 
                          normal::Vector{Float64}, 
                          tolerance::Float64=1.0)

Backward-compatible convenience function that automatically manages surface node caching.
This version maintains the same interface as the original function.

Parameters:
- `grid::Grid`: Computational mesh
- `point::Vector{Float64}`: A point on the plane [x, y, z]
- `normal::Vector{Float64}`: Normal vector to the plane [nx, ny, nz]  
- `tolerance::Float64`: Distance tolerance for node selection

Returns:
- `Set{Int}`: Set of surface node IDs that lie on the specified plane
"""
function select_nodes_by_plane(
    grid::Grid,
    point::Vector{Float64},
    normal::Vector{Float64},
    tolerance::Float64 = 1.0,
)
    # Get or create cache automatically
    cache = get_or_create_cache(grid)

    # Use optimized surface-only selection
    return select_surface_nodes_by_plane(cache, point, normal, tolerance)
end

"""
    select_nodes_by_circle(grid::Grid, 
                           center::Vector{Float64}, 
                           normal::Vector{Float64}, 
                           radius::Float64, 
                           tolerance::Float64=1.0)

Backward-compatible convenience function that automatically manages surface node caching.
This version maintains the same interface as the original function.

Parameters:
- `grid::Grid`: Computational mesh
- `center::Vector{Float64}`: Center of the circle [x, y, z]
- `normal::Vector{Float64}`: Normal vector to the plane containing the circle [nx, ny, nz]
- `radius::Float64`: Radius of the circle
- `tolerance::Float64`: Distance tolerance for node selection

Returns:
- `Set{Int}`: Set of surface node IDs that lie within the circular region
"""
function select_nodes_by_circle(
    grid::Grid,
    center::Vector{Float64},
    normal::Vector{Float64},
    radius::Float64,
    tolerance::Float64 = 1.0,
)
    # Get or create cache automatically
    cache = get_or_create_cache(grid)

    # Use optimized surface-only selection
    return select_surface_nodes_by_circle(cache, center, normal, radius, tolerance)
end

"""
    clear_surface_cache!()

Clears all cached surface node data. Useful for memory management when working
with many different grids or when grids are modified.

Returns:
- `nothing`
"""
function clear_surface_cache!()
    empty!(GRID_CACHE_STORAGE)
    println("Surface node cache cleared")
end

"""
    precompute_surface_nodes!(grid::Grid)

Explicitly pre-compute surface nodes for a grid. This can be useful when you want
to front-load the computation cost rather than having it happen on first access.

Parameters:
- `grid::Grid`: Computational mesh

Returns:
- `SurfaceNodeCache`: The computed cache object
"""
function precompute_surface_nodes!(grid::Grid)
    return get_or_create_cache(grid)
end
