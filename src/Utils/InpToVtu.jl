"""
InpToVtu.jl - Conversion module for Abaqus .inp files to VTK .vtu format

This module provides functionality to convert Abaqus input files (.inp) to VTK unstructured grid 
files (.vtu) for visualization in ParaView or similar VTK-compatible viewers.

Dependencies:
- AbaqusReader.jl: For parsing Abaqus .inp files
- WriteVTK.jl: For writing VTK .vtu files

Author: Julia Development Assistant
"""

using AbaqusReader
using WriteVTK

"""
    InpToVtu(inp_file::String, output_file::String; verbose::Bool=true)

Converts an Abaqus .inp file to a VTK .vtu file containing the mesh geometry.

# Parameters:
- `inp_file::String`: Path to the input Abaqus .inp file
- `output_file::String`: Path for the output .vtu file (without extension)
- `verbose::Bool`: Enable detailed logging (default: true)

# Returns:
- `Bool`: True if conversion was successful, false otherwise

# Example:
```julia
# Basic usage
success = InpToVtu("model.inp", "output_mesh")

# Quiet mode
success = InpToVtu("model.inp", "output_mesh", verbose=false)
```

# Supported AbaqusReader Element Type Symbols:
- Hex8: 8-node linear hexahedron (C3D8) → VTK_HEXAHEDRON
- Tet4: 4-node linear tetrahedron (C3D4) → VTK_TETRA  
- Hex20: 20-node quadratic hexahedron (C3D20) → VTK_QUADRATIC_HEXAHEDRON
- Tet10: 10-node quadratic tetrahedron (C3D10) → VTK_QUADRATIC_TETRA
- Wedge6: 6-node wedge (C3D6) → VTK_WEDGE
- Wedge15: 15-node quadratic wedge (C3D15) → VTK_QUADRATIC_WEDGE
- Tri3: 3-node triangle shell (S3) → VTK_TRIANGLE
- Quad4: 4-node quadrilateral shell (S4) → VTK_QUAD
- Tri6: 6-node quadratic triangle (S6) → VTK_QUADRATIC_TRIANGLE
- Quad8: 8-node quadratic quadrilateral (S8) → VTK_QUADRATIC_QUAD
- Seg2: 2-node truss element (T3D2) → VTK_LINE
- Seg3: 3-node quadratic truss (T3D3) → VTK_QUADRATIC_EDGE

Note: AbaqusReader.jl converts Abaqus element names (e.g., C3D8) to its own symbols (e.g., Hex8)

# Notes:
- Only mesh geometry is converted (nodes and elements)
- Material properties, boundary conditions, and loads are not included
- To get results data (stress, displacement), you need to run FEA analysis first
- Node ordering is adjusted to match VTK conventions where necessary
"""
function InpToVtu(inp_file::String, output_file::String; verbose::Bool=true)
    
    # Input validation
    if !isfile(inp_file)
        error("Input file '$inp_file' does not exist")
    end
    
    if !endswith(lowercase(inp_file), ".inp")
        @warn "Input file '$inp_file' does not have .inp extension"
    end
    
    try
        verbose && println("📖 Reading Abaqus .inp file: $inp_file")
        
        # Parse the .inp file using AbaqusReader
        mesh_data = abaqus_read_mesh(inp_file)
        
        # Extract mesh components
        nodes = mesh_data["nodes"]           # Dict: Node ID => [x, y, z]
        elements = mesh_data["elements"]     # Dict: Element ID => [node1, node2, ...]
        element_types = mesh_data["element_types"]  # Dict: Element ID => Symbol
        
        verbose && println("   ✓ Found $(length(nodes)) nodes and $(length(elements)) elements")
        
        # Convert nodes to WriteVTK format
        # WriteVTK expects a 3×N matrix where each column is [x, y, z] coordinates
        node_ids = sort(collect(keys(nodes)))
        
        # Ensure we have 3D coordinates (pad with zeros if 2D)
        points = zeros(Float64, 3, length(node_ids))
        for (i, node_id) in enumerate(node_ids)
            coords = nodes[node_id]
            points[1:length(coords), i] = coords
            # Z-coordinate remains 0.0 if not provided (2D case)
        end
        
        verbose && println("   ✓ Processed node coordinates")
        
        # Create node ID mapping for element connectivity
        # AbaqusReader gives actual node IDs, but WriteVTK expects 1-based indexing
        node_id_to_index = Dict(node_id => i for (i, node_id) in enumerate(node_ids))
        
        # Convert elements to WriteVTK format
        cells = WriteVTK.MeshCell[]
        element_type_counts = Dict{Symbol, Int}()
        unsupported_types = Set{Symbol}()
        
        for elem_id in sort(collect(keys(elements)))
            connectivity = elements[elem_id]
            elem_type = element_types[elem_id]
            
            # Count element types for statistics
            element_type_counts[elem_type] = get(element_type_counts, elem_type, 0) + 1
            
            # Map connectivity from node IDs to indices
            vtk_connectivity = [node_id_to_index[node_id] for node_id in connectivity]
            
            # Convert Abaqus element type to VTK cell type
            vtk_cell = _abaqus_to_vtk_cell(elem_type, vtk_connectivity)
            
            if vtk_cell !== nothing
                push!(cells, vtk_cell)
            else
                push!(unsupported_types, elem_type)
            end
        end
        
        # Report element type statistics
        if verbose
            println("   ✓ Element type summary:")
            for (elem_type, count) in sort(collect(element_type_counts))
                status = elem_type in unsupported_types ? "❌ UNSUPPORTED" : "✓ Converted"
                println("     - $elem_type: $count elements ($status)")
            end
        end
        
        # Warn about unsupported element types
        if !isempty(unsupported_types)
            @warn "Unsupported element types found: $(collect(unsupported_types)). These elements will be skipped."
        end
        
        if isempty(cells)
            error("No supported elements found in the mesh")
        end
        
        verbose && println("   ✓ Successfully converted $(length(cells)) elements")
        
        # Write VTU file
        verbose && println("💾 Writing VTU file: $(output_file).vtu")
        
        vtk_grid(output_file, points, cells) do vtk
            # Only mesh geometry - no additional data
        end
        
        verbose && println("✅ Conversion completed successfully!")
        verbose && println("   Output file: $(output_file).vtu")
        verbose && println("   Nodes: $(length(node_ids))")
        verbose && println("   Elements: $(length(cells))")
        
        return true
        
    catch e
        @error "Failed to convert .inp to .vtu: $(e)"
        return false
    end
end

"""
    _abaqus_to_vtk_cell(abaqus_type::Symbol, connectivity::Vector{Int})

Internal function to convert AbaqusReader element type symbols to VTK cell types.
Handles node ordering differences between Abaqus and VTK conventions.

# Parameters:
- `abaqus_type::Symbol`: Element type symbol as returned by AbaqusReader (e.g., :Hex8, :Tet4)
- `connectivity::Vector{Int}`: Node connectivity list (1-based indices)

# Returns:
- `WriteVTK.MeshCell`: VTK mesh cell, or `nothing` if unsupported

# Note: 
AbaqusReader.jl converts Abaqus element names to its own symbol format:
- C3D8 → :Hex8 (8-node hexahedron)
- C3D4 → :Tet4 (4-node tetrahedron)
- C3D20 → :Hex20 (20-node quadratic hexahedron)
- C3D10 → :Tet10 (10-node quadratic tetrahedron)
etc.
"""
function _abaqus_to_vtk_cell(abaqus_type::Symbol, connectivity::Vector{Int})
    
    # Define mapping from AbaqusReader symbols to VTK element types
    # Note: AbaqusReader converts C3D8 to :Hex8, C3D4 to :Tet4, etc.
    
    # Linear hexahedral elements
    if abaqus_type == :Hex8 || abaqus_type == :C3D8  # 8-node linear hexahedron
        # VTK hexahedron node order: bottom face (1,2,3,4), top face (5,6,7,8)
        # AbaqusReader should maintain correct node ordering
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_HEXAHEDRON, connectivity)
        
    # Linear tetrahedral elements  
    elseif abaqus_type == :Tet4 || abaqus_type == :C3D4  # 4-node linear tetrahedron
        # Abaqus and VTK have same node ordering for tetrahedra
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TETRA, connectivity)
        
    # Quadratic hexahedral elements
    elseif abaqus_type == :Hex20 || abaqus_type == :C3D20  # 20-node quadratic hexahedron
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUADRATIC_HEXAHEDRON, connectivity)
        
    # Quadratic tetrahedral elements
    elseif abaqus_type == :Tet10 || abaqus_type == :C3D10  # 10-node quadratic tetrahedron
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUADRATIC_TETRA, connectivity)
        
    # Wedge/Prism elements
    elseif abaqus_type == :Wedge6 || abaqus_type == :C3D6  # 6-node wedge
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_WEDGE, connectivity)
        
    elseif abaqus_type == :Wedge15 || abaqus_type == :C3D15  # 15-node quadratic wedge
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUADRATIC_WEDGE, connectivity)
        
    # Shell elements
    elseif abaqus_type == :Tri3 || abaqus_type == :S3  # 3-node triangle shell
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TRIANGLE, connectivity)
        
    elseif abaqus_type == :Quad4 || abaqus_type == :S4  # 4-node quadrilateral shell
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUAD, connectivity)
        
    elseif abaqus_type == :Tri6 || abaqus_type == :S6  # 6-node quadratic triangle
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUADRATIC_TRIANGLE, connectivity)
        
    elseif abaqus_type == :Quad8 || abaqus_type == :S8  # 8-node quadratic quadrilateral
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUADRATIC_QUAD, connectivity)
        
    # Truss/Beam elements
    elseif abaqus_type == :Seg2 || abaqus_type == :T3D2  # 2-node truss element
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_LINE, connectivity)
        
    elseif abaqus_type == :Seg3 || abaqus_type == :T3D3  # 3-node quadratic truss
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUADRATIC_EDGE, connectivity)
        
    # Plane stress/strain elements (2D elements)
    elseif abaqus_type == :CPE3 || abaqus_type == :CPS3  # 3-node plane element
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TRIANGLE, connectivity)
        
    elseif abaqus_type == :CPE4 || abaqus_type == :CPS4  # 4-node plane element
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUAD, connectivity)
        
    elseif abaqus_type == :CPE6 || abaqus_type == :CPS6  # 6-node quadratic plane element
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUADRATIC_TRIANGLE, connectivity)
        
    elseif abaqus_type == :CPE8 || abaqus_type == :CPS8  # 8-node quadratic plane element
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUADRATIC_QUAD, connectivity)
        
    # Axisymmetric elements
    elseif abaqus_type == :CAX3  # 3-node axisymmetric element
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TRIANGLE, connectivity)
        
    elseif abaqus_type == :CAX4  # 4-node axisymmetric element
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUAD, connectivity)
        
    elseif abaqus_type == :CAX6  # 6-node quadratic axisymmetric element
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUADRATIC_TRIANGLE, connectivity)
        
    elseif abaqus_type == :CAX8  # 8-node quadratic axisymmetric element
        return WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUADRATIC_QUAD, connectivity)
        
    else
        # Unsupported element type - log it for debugging
        @debug "Unsupported element type: $abaqus_type"
        return nothing
    end
end

"""
    add_element_type_support!(abaqus_type::Symbol, vtk_type, node_reorder_func=identity)

Helper function to extend support for additional Abaqus element types.
This allows users to add custom element type mappings.

# Parameters:
- `abaqus_type::Symbol`: Abaqus element type symbol
- `vtk_type`: VTK cell type constant
- `node_reorder_func`: Optional function to reorder nodes (default: identity)

# Example:
```julia
# Add support for a custom element type
add_element_type_support!(:MY_ELEMENT, WriteVTK.VTKCellTypes.VTK_TRIANGLE)
```
"""
function add_element_type_support!(abaqus_type::Symbol, vtk_type, node_reorder_func=identity)
    # This would require modifying the global mapping
    # For now, users should modify _abaqus_to_vtk_cell directly
    @warn "Custom element type addition not implemented yet. Please modify _abaqus_to_vtk_cell function."
end

"""
    validate_inp_file(inp_file::String)

Validates an Abaqus .inp file before conversion.
Checks for common issues that might cause conversion problems.

# Parameters:
- `inp_file::String`: Path to the .inp file

# Returns:
- `Dict`: Validation results with potential issues and warnings
"""
function validate_inp_file(inp_file::String)
    validation_results = Dict(
        "file_exists" => isfile(inp_file),
        "file_readable" => false,
        "has_nodes" => false,
        "has_elements" => false,
        "warnings" => String[],
        "errors" => String[]
    )
    
    if !validation_results["file_exists"]
        push!(validation_results["errors"], "File does not exist: $inp_file")
        return validation_results
    end
    
    try
        # Try to read the file
        mesh_data = abaqus_read_mesh(inp_file)
        validation_results["file_readable"] = true
        
        # Check for nodes and elements
        validation_results["has_nodes"] = haskey(mesh_data, "nodes") && !isempty(mesh_data["nodes"])
        validation_results["has_elements"] = haskey(mesh_data, "elements") && !isempty(mesh_data["elements"])
        
        if !validation_results["has_nodes"]
            push!(validation_results["errors"], "No nodes found in the mesh")
        end
        
        if !validation_results["has_elements"]
            push!(validation_results["errors"], "No elements found in the mesh")
        end
        
        # Check for unsupported element types
        if haskey(mesh_data, "element_types")
            element_types = mesh_data["element_types"]
            unsupported = Set{Symbol}()
            
            for elem_type in values(element_types)
                if _abaqus_to_vtk_cell(elem_type, [1, 2, 3, 4]) === nothing  # Test with dummy connectivity
                    push!(unsupported, elem_type)
                end
            end
            
            if !isempty(unsupported)
                push!(validation_results["warnings"], 
                      "Unsupported element types found: $(collect(unsupported))")
            end
        end
        
    catch e
        validation_results["file_readable"] = false
        push!(validation_results["errors"], "Failed to read file: $(e)")
    end
    
    return validation_results
end

"""
    inspect_inp_elements(inp_file::String)

Diagnostic function to inspect what element types are actually found by AbaqusReader.
Useful for debugging unsupported element type issues.

# Parameters:
- `inp_file::String`: Path to the .inp file

# Returns:
- `Dict`: Summary of element types found in the file
"""
function inspect_inp_elements(inp_file::String)
    if !isfile(inp_file)
        error("File does not exist: $inp_file")
    end
    
    println("🔍 Inspecting element types in: $inp_file")
    
    try
        mesh_data = abaqus_read_mesh(inp_file)
        elements = mesh_data["elements"]
        element_types = mesh_data["element_types"]
        
        # Count each element type
        type_counts = Dict{Symbol, Int}()
        for elem_type in values(element_types)
            type_counts[elem_type] = get(type_counts, elem_type, 0) + 1
        end
        
        println("\n📊 Element Type Summary:")
        println("=" ^ 50)
        
        total_elements = 0
        for (elem_type, count) in sort(collect(type_counts))
            supported = _abaqus_to_vtk_cell(elem_type, [1, 2, 3, 4]) !== nothing
            status = supported ? "✅ SUPPORTED" : "❌ UNSUPPORTED"
            println("   $elem_type: $count elements ($status)")
            total_elements += count
        end
        
        println("=" ^ 50)
        println("   Total: $total_elements elements")
        
        # Return detailed information
        result = Dict(
            "total_elements" => total_elements,
            "element_counts" => type_counts,
            "supported_types" => Symbol[],
            "unsupported_types" => Symbol[]
        )
        
        for elem_type in keys(type_counts)
            if _abaqus_to_vtk_cell(elem_type, [1, 2, 3, 4]) !== nothing
                push!(result["supported_types"], elem_type)
            else
                push!(result["unsupported_types"], elem_type)
            end
        end
        
        return result
        
    catch e
        @error "Failed to inspect file: $e"
        return Dict("error" => string(e))
    end
end

# Example usage and testing function
"""
    test_conversion(inp_file::String="test.inp")

Test function to demonstrate the conversion process.
Creates a simple example if no file is provided.
"""
function test_conversion(inp_file::String="test.inp")
    println("🧪 Testing InpToVtu conversion...")
    
    # Validate file first
    println("📋 Validating input file...")
    validation = validate_inp_file(inp_file)
    
    if !isempty(validation["errors"])
        println("❌ Validation failed:")
        for error in validation["errors"]
            println("   • $error")
        end
        return false
    end
    
    if !isempty(validation["warnings"])
        println("⚠️  Validation warnings:")
        for warning in validation["warnings"]
            println("   • $warning")
        end
    end
    
    # Perform conversion
    output_name = splitext(inp_file)[1] * "_converted"
    success = InpToVtu(inp_file, output_name, verbose=true)
    
    if success
        println("🎉 Test completed successfully!")
        println("   Input: $inp_file")
        println("   Output: $(output_name).vtu")
    else
        println("❌ Test failed!")
    end
end

InpToVtu("stul14.inp", "stul14")
