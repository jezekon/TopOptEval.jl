# Load the ReadVTK package
using ReadVTK
using StaticArrays  # For creating Ferrite nodes later

# Set the path to your VTU file
vtu_file = "data/cantilever_beam_volume_mesh.vtu"

# Open the VTU file
vtk = VTKFile(vtu_file)
println("Successfully opened VTU file: $(typeof(vtk))")

# Extract node coordinates
points = get_points(vtk)
num_nodes = length(points)
println("Number of nodes: $num_nodes")

# Print first few nodes to verify
println("First 3 nodes:")
for i in 1:min(3, num_nodes)
    println("Node $i: $(points[i])")
end

# Extract cell types
cell_types = get_cell_types(vtk)
println("Number of cells: $(length(cell_types))")

# Count tetrahedral elements (VTK_TETRA = 10)
tetra_indices = findall(cell_types .== 10)
println("Number of tetrahedral elements: $(length(tetra_indices))")

# Extract cell connectivity
cell_connectivity = get_cell_connectivity(vtk)
println("Connectivity array length: $(length(cell_connectivity))")

# Extract cell offsets
offsets = get_cell_offsets(vtk)
println("Offsets array length: $(length(offsets))")

# Extract the first few tetrahedral elements
println("\nFirst 3 tetrahedral elements:")
for i in 1:min(3, length(tetra_indices))
    # Get the index of this tetrahedral element
    tetra_idx = tetra_indices[i]
    
    # Get start and end indices in the connectivity array
    start_idx = tetra_idx == 1 ? 1 : offsets[tetra_idx-1] + 1
    end_idx = offsets[tetra_idx]
    
    # Extract the node indices for this tetrahedron (convert to 1-based for Julia)
    node_indices = cell_connectivity[start_idx:end_idx] .+ 1
    
    println("Element $i (VTK index $(tetra_indices[i])): Nodes $node_indices")
end

# Find nodes at x ≈ 0 (fixed boundary)
tol = 1e-6
fixed_nodes = Int[]
for i in 1:num_nodes
    if abs(points[i][1]) < tol
        push!(fixed_nodes, i)
    end
end
println("\nNumber of fixed nodes (x ≈ 0): $(length(fixed_nodes))")
if !isempty(fixed_nodes)
    println("First few fixed nodes: $(fixed_nodes[1:min(5, length(fixed_nodes))])")
end

# Find nodes at x ≈ 60 (loaded boundary)
load_nodes = Int[]
for i in 1:num_nodes
    if abs(points[i][1] - 60.0) < tol
        push!(load_nodes, i)
    end
end
println("\nNumber of load nodes (x ≈ 60): $(length(load_nodes))")
if !isempty(load_nodes)
    println("First few load nodes: $(load_nodes[1:min(5, length(load_nodes))])")
end

println("\nNode and element data extraction complete!")

# The extracted data can now be used to create a Ferrite Grid:
# 
# using Ferrite
# 
# # Create Ferrite nodes
# nodes = [Node(point) for point in points]
# 
# # Create Ferrite tetrahedral elements
# elements = Tetrahedron[]
# for i in tetra_indices
#     # Get start and end indices in the connectivity array
#     start_idx = i == 1 ? 1 : offsets[i-1] + 1
#     end_idx = offsets[i]
#     
#     # Extract node indices (convert to 1-based)
#     node_indices = cell_connectivity[start_idx:end_idx] .+ 1
#     
#     # Create a Ferrite Tetrahedron element
#     push!(elements, Tetrahedron((node_indices...)))
# end
# 
# # Create the Ferrite Grid
# grid = Grid(elements, nodes)