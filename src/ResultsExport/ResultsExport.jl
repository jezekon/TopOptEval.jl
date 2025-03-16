module ResultsExport

using Ferrite
using WriteVTK
using LinearAlgebra  # Add this import for eigvals function
using Tensors        # Add this for stress tensor operations

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

"""
    export_results(stress_field::Dict{Int64, Vector{T}},
                  dh::DofHandler,
                  output_file::String) where {T <: SymmetricTensor{2, 3}}

Exports stress tensor field results to a VTK file for visualization in ParaView.
Converts stress tensors to von Mises stress and principal stresses for easier visualization.

Parameters:
- `stress_field`: Dictionary mapping cell IDs to vectors of stress tensors at quadrature points
- `dh`: DofHandler used in the analysis
- `output_file`: Path to the output file (without extension)

Returns:
- nothing
"""
function export_results(stress_field::Dict{Int64, Vector{T}},
                       dh::DofHandler,
                       output_file::String) where {T <: SymmetricTensor{2, 3}}
    println("Exporting stress results to $output_file...")
    
    # Create a VTK file
    vtk_file = VTKGridFile(output_file, dh.grid) do vtk
        # Calculate von Mises stress for each cell
        von_mises_stresses = zeros(getncells(dh.grid))
        principal_stress_1 = zeros(getncells(dh.grid))
        principal_stress_3 = zeros(getncells(dh.grid))
        
        for (cell_id, stress_tensors) in stress_field
            # Average the stress tensors at quadrature points if there are multiple
            if !isempty(stress_tensors)
                avg_stress = sum(stress_tensors) / length(stress_tensors)
                
                # Calculate von Mises stress
                von_mises_stresses[cell_id] = sqrt(3/2 * (dev(avg_stress) ⊡ dev(avg_stress)))
                
                # Calculate principal stresses (eigenvalues)
                eigvals = sort(LinearAlgebra.eigvals(avg_stress))
                principal_stress_1[cell_id] = eigvals[3]  # Maximum principal stress
                principal_stress_3[cell_id] = eigvals[1]  # Minimum principal stress
            end
        end
        
        # Write the von Mises stress field to the VTK file
        write_cell_data(vtk, von_mises_stresses, "von_Mises_stress")
        write_cell_data(vtk, principal_stress_1, "principal_stress_max")
        write_cell_data(vtk, principal_stress_3, "principal_stress_min")
    end
    
    println("Stress results exported successfully to $(output_file).vtu")
end

end # module
