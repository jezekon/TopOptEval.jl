module MeshImport

using Ferrite
using FerriteGmsh  # Required for parsing GMSH (.msh) files
using ReadVTK      # Required for parsing VTK XML formats (.vtu files)
export import_mesh
    
"""
    import_mesh(mesh_file::String)

Imports a mesh file and returns the Ferrite Grid object.
Currently supports GMSH (.msh) and VTK XML UnstructuredGrid (.vtu) files.

Parameters:
- `mesh_file`: Path to the mesh file (.msh or .vtu)

Returns:
- A Ferrite Grid object
"""
function import_mesh(mesh_file::String)
    # Check file extension
    ext = lowercase(splitext(mesh_file)[2])
    
    if ext == ".msh"
        println("Importing GMSH mesh from $mesh_file...")
        
        # Import the GMSH .msh file using FerriteGmsh
        grid = FerriteGmsh.togrid(mesh_file)
        
    elseif ext == ".vtu"
        println("Importing VTU mesh from $mesh_file...")
        
        # Read the VTU file using ReadVTK
        vtk_file = ReadVTK.VTKFile(mesh_file)
        
        # Extract coordinates from points
        points = ReadVTK.get_points(vtk_file)
        
        # Extract cells
        vtk_cells = ReadVTK.get_cells(vtk_file)
        
        # Create Ferrite nodes from points (transpose to iterate over columns)
        nodes = [Ferrite.Node(Vec{3}(points[:, i])) for i in 1:size(points, 2)]
        
        # Create Ferrite cells from VTK cells
        ferrite_cells = Ferrite.AbstractCell[]
        
        # Process connectivity, offsets, and types from vtk_cells
        connectivity = vtk_cells.connectivity
        offsets = vtk_cells.offsets
        types = vtk_cells.types
        
        # Create start offsets for cell connectivity indexing
        start_indices = vcat(1, offsets[1:end-1] .+ 1)
        
        for i in 1:length(types)
            # Get connectivity indices for this cell
            conn_indices = start_indices[i]:offsets[i]
            cell_conn = connectivity[conn_indices]
            
            # Map VTK cell types to Ferrite cell types
            # Reference: https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
            vtk_type = types[i]
            
            # VTK_TETRA = 10
            if vtk_type == 10
                push!(ferrite_cells, Ferrite.Tetrahedron(Tuple(cell_conn)))
            # VTK_HEXAHEDRON = 12
            elseif vtk_type == 12
                push!(ferrite_cells, Ferrite.Hexahedron(Tuple(cell_conn)))
            # VTK_TRIANGLE = 5
            elseif vtk_type == 5
                push!(ferrite_cells, Ferrite.Triangle(Tuple(cell_conn)))
            # VTK_QUAD = 9
            elseif vtk_type == 9
                push!(ferrite_cells, Ferrite.Quadrilateral(Tuple(cell_conn)))
            # VTK_LINE = 3
            elseif vtk_type == 3
                push!(ferrite_cells, Ferrite.Line(Tuple(cell_conn)))
            else
                @warn "Unsupported VTK cell type: $vtk_type, skipping"
            end
        end
        
        # Create Ferrite grid from nodes and cells
        grid = Ferrite.Grid(ferrite_cells, nodes)
        
        # Try to import cell data if present
        try
            cell_data = ReadVTK.get_cell_data(vtk_file)
            
            # Look for potential cell sets in cell data
            # Common names for cell entity IDs in different formats
            potential_cellset_names = ["CellEntityIds", "element_ids", "gmsh:physical", "ElementId"]
            
            for name in potential_cellset_names
                if name in ReadVTK.keys(cell_data)
                    entity_ids_array = ReadVTK.get_data(cell_data[name])
                    unique_ids = unique(entity_ids_array)
                    
                    for id in unique_ids
                        cells_in_set = findall(entity_ids_array .== id)
                        if !isempty(cells_in_set)
                            Ferrite.addcellset!(grid, "cellset_$id", Set(cells_in_set))
                            println("  Added cellset_$id with $(length(cells_in_set)) cells")
                        end
                    end
                    
                    # Break after finding and processing the first valid cell set data
                    break
                end
            end
        catch e
            @warn "Could not import cell data: $e"
        end
        
    else
        error("Unsupported mesh format: $ext. Only .msh and .vtu formats are supported.")
    end
    
    println("Mesh imported successfully: $(Ferrite.getnnodes(grid)) nodes, $(Ferrite.getncells(grid)) elements")
    
    return grid
end


end # module
