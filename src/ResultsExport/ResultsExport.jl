module ResultsExport

using Ferrite
using WriteVTK

export export_results

"""
    export_results(displacements::Vector{Float64}, 
                   dh::DofHandler, 
                   output_file::String)

Exports FEM analysis results to a VTK file for visualization in ParaView.

Parameters:
- `displacements`: Solution vector containing displacements
- `dh`: DofHandler used in the analysis
- `output_file`: Path to the output file (without extension)

Returns:
- nothing
"""
function export_results(displacements::Vector{Float64}, dh::DofHandler, output_file::String)
    println("Exporting results to $output_file...")
    
    # Create a VTK file to visualize the results
    vtk_file = VTKGridFile(output_file, dh)
    
    # Add displacement field to the VTK file
    write_solution(vtk_file, dh, displacements)
    
    # Write the results
    close(vtk_file)
    println("Results exported successfully to $(output_file).vtu")
end

end # module
