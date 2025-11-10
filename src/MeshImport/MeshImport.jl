module MeshImport

using Ferrite
using FerriteGmsh  # Required for parsing GMSH (.msh) files
using ReadVTK      # Required for parsing VTK XML formats (.vtu files)
export import_mesh, extract_cell_density

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
        nodes = [Ferrite.Node(Vec{3}(points[:, i])) for i = 1:size(points, 2)]

        # Process connectivity, offsets, and types from vtk_cells
        connectivity = vtk_cells.connectivity
        offsets = vtk_cells.offsets
        types = vtk_cells.types

        # Create start offsets for cell connectivity indexing
        start_indices = vcat(1, offsets[1:(end-1)] .+ 1)

        # Group cells by type to create homogeneous grids
        tetrahedron_cells = Ferrite.Tetrahedron[]
        hexahedron_cells = Ferrite.Hexahedron[]
        triangle_cells = Ferrite.Triangle[]
        quadrilateral_cells = Ferrite.Quadrilateral[]
        line_cells = Ferrite.Line[]

        # Keep track of the dominant cell type
        cell_counts = Dict{Int,Int}()

        for i = 1:length(types)
            # Get connectivity indices for this cell
            conn_indices = start_indices[i]:offsets[i]
            cell_conn = connectivity[conn_indices]

            # Update cell type count
            vtk_type = types[i]
            cell_counts[vtk_type] = get(cell_counts, vtk_type, 0) + 1

            # VTK_TETRA = 10
            if vtk_type == 10
                push!(tetrahedron_cells, Ferrite.Tetrahedron(Tuple(cell_conn)))
                # VTK_HEXAHEDRON = 12
            elseif vtk_type == 12
                push!(hexahedron_cells, Ferrite.Hexahedron(Tuple(cell_conn)))
                # VTK_TRIANGLE = 5
            elseif vtk_type == 5
                push!(triangle_cells, Ferrite.Triangle(Tuple(cell_conn)))
                # VTK_QUAD = 9
            elseif vtk_type == 9
                push!(quadrilateral_cells, Ferrite.Quadrilateral(Tuple(cell_conn)))
                # VTK_LINE = 3
            elseif vtk_type == 3
                push!(line_cells, Ferrite.Line(Tuple(cell_conn)))
            else
                @warn "Unsupported VTK cell type: $vtk_type, skipping"
            end
        end

        # Create a grid with the most common cell type
        if !isempty(cell_counts)
            dominant_type = argmax(cell_counts)
            println("  Dominant cell type: $dominant_type")

            if dominant_type == 10 && !isempty(tetrahedron_cells)
                grid = Ferrite.Grid(tetrahedron_cells, nodes)
                println(
                    "  Created grid with $(length(tetrahedron_cells)) Tetrahedron cells",
                )
            elseif dominant_type == 12 && !isempty(hexahedron_cells)
                grid = Ferrite.Grid(hexahedron_cells, nodes)
                println("  Created grid with $(length(hexahedron_cells)) Hexahedron cells")
            elseif dominant_type == 5 && !isempty(triangle_cells)
                grid = Ferrite.Grid(triangle_cells, nodes)
                println("  Created grid with $(length(triangle_cells)) Triangle cells")
            elseif dominant_type == 9 && !isempty(quadrilateral_cells)
                grid = Ferrite.Grid(quadrilateral_cells, nodes)
                println(
                    "  Created grid with $(length(quadrilateral_cells)) Quadrilateral cells",
                )
            elseif dominant_type == 3 && !isempty(line_cells)
                grid = Ferrite.Grid(line_cells, nodes)
                println("  Created grid with $(length(line_cells)) Line cells")
            else
                error("No supported cell types found in the mesh")
            end
        else
            error("No cells found in the mesh")
        end

        # Try to import cell data if present
        try
            cell_data = ReadVTK.get_cell_data(vtk_file)

            # Look for potential cell sets in cell data
            # Common names for cell entity IDs in different formats
            potential_cellset_names =
                ["CellEntityIds", "element_ids", "gmsh:physical", "ElementId"]

            for name in potential_cellset_names
                if name in ReadVTK.keys(cell_data)
                    entity_ids_array = ReadVTK.get_data(cell_data[name])
                    unique_ids = unique(entity_ids_array)

                    for id in unique_ids
                        cells_in_set = findall(entity_ids_array .== id)
                        if !isempty(cells_in_set)
                            Ferrite.addcellset!(grid, "cellset_$id", Set(cells_in_set))
                            println(
                                "  Added cellset_$id with $(length(cells_in_set)) cells",
                            )
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

    println(
        "Mesh imported successfully: $(Ferrite.getnnodes(grid)) nodes, $(Ferrite.getncells(grid)) elements",
    )

    return grid
end

"""
    extract_cell_density(mesh_file::String)

Extracts density/volume fraction data from a VTU file.

Parameters:
- `mesh_file`: Path to the mesh file (.vtu)

Returns:
- Vector of density values for each cell
"""
function extract_cell_density(mesh_file::String)
    # Check file extension
    ext = lowercase(splitext(mesh_file)[2])

    if ext != ".vtu"
        error("Density extraction is only supported for VTU files")
    end

    density_data = nothing

    try
        # Read the VTU file
        vtk_file = ReadVTK.VTKFile(mesh_file)

        # Try to get cell data
        vtk_cell_data = ReadVTK.get_cell_data(vtk_file)

        # Look for density data (try common names)
        density_field_names =
            ["density", "rho", "Density", "DENSITY", "volfrac", "VolFrac", "vol_frac"]

        for name in density_field_names
            if name in ReadVTK.keys(vtk_cell_data)
                density_array = ReadVTK.get_data(vtk_cell_data[name])
                density_data = collect(density_array)
                println("  Extracted density data from field '$name'")
                break
            end
        end
    catch e
        @warn "Could not extract density data: $e"
    end

    if density_data === nothing
        error("No density data found in the mesh file")
    end

    return density_data
end

end # module
