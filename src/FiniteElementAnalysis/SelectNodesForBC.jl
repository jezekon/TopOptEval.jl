"""
    select_nodes_by_plane(grid::Grid, 
                          point::Vector{Float64}, 
                          normal::Vector{Float64}, 
                          tolerance::Float64=1e-6)

Selects nodes that lie on a plane defined by a point and normal vector.

Parameters:
- `grid`: Computational mesh
- `point`: A point on the plane [x, y, z]
- `normal`: Normal vector to the plane [nx, ny, nz]
- `tolerance`: Distance tolerance for node selection

Returns:
- Set of node IDs that lie on the plane
"""
function select_nodes_by_plane(grid::Grid, 
                               point::Vector{Float64}, 
                               normal::Vector{Float64}, 
                               tolerance::Float64=1.)
                               # tolerance::Float64=1e-2)
    # Normalize the normal vector
    unit_normal = normal / norm(normal)
    
    # Extract number of nodes
    num_nodes = getnnodes(grid)
    selected_nodes = Set{Int}()
    
    # Check each node
    for node_id in 1:num_nodes
        coord = grid.nodes[node_id].x
        
        # Calculate distance from point to plane: d = (p - p0) · n
        dist = abs(dot(coord - point, unit_normal))
        
        # If distance is within tolerance, node is on plane
        if dist < tolerance
            push!(selected_nodes, node_id)
        end
    end
    
    println("Selected $(length(selected_nodes)) nodes on the specified plane")
    return selected_nodes
end

"""
    select_nodes_by_circle(grid::Grid, 
                           center::Vector{Float64}, 
                           normal::Vector{Float64}, 
                           radius::Float64, 
                           tolerance::Float64=1e-6)

Selects nodes that lie on a circular region defined by center, normal and radius.

Parameters:
- `grid`: Computational mesh
- `center`: Center of the circle [x, y, z]
- `normal`: Normal vector to the plane containing the circle [nx, ny, nz]
- `radius`: Radius of the circle
- `tolerance`: Distance tolerance for node selection

Returns:
- Set of node IDs that lie on the circular region
"""
function select_nodes_by_circle(grid::Grid, 
                                center::Vector{Float64}, 
                                normal::Vector{Float64}, 
                                radius::Float64, 
                                tolerance::Float64=1.)
                                # tolerance::Float64=1e-2)
    # First, get nodes on the plane
    nodes_on_plane = select_nodes_by_plane(grid, center, normal, tolerance)
    
    # Normalize the normal vector
    unit_normal = normal / norm(normal)
    
    # Initialize set for nodes in circle
    nodes_in_circle = Set{Int}()
    
    # Check which nodes are within the circle radius
    for node_id in nodes_on_plane
        coord = grid.nodes[node_id].x
        
        # Project the vector from center to node onto the plane
        v = coord - center
        projection = v - dot(v, unit_normal) * unit_normal
        
        # Calculate distance from center in the plane
        dist = norm(projection)
        
        # If distance is less than radius, node is in the circle
        if dist <= radius + tolerance
            push!(nodes_in_circle, node_id)
        end
    end
    
    println("Selected $(length(nodes_in_circle)) nodes in the circular region")
    return nodes_in_circle
end
