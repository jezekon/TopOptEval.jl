module ResultsExport

using Ferrite
using WriteVTK
using LinearAlgebra  # Add this import for eigvals function
using Tensors        # Add this for stress tensor operations

export export_results, export_boundary_conditions

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

"""
    export_boundary_conditions(grid::Grid, dh::DofHandler, fixed_nodes::Set{Int}, 
                              force_nodes::Set{Int}, output_file::String)

Exportuje síť s označenými okrajovými podmínkami do VTU souboru.

Parametry:
- `grid`: Výpočetní síť
- `dh`: DofHandler ze setupu problému
- `fixed_nodes`: Množina ID uzlů s pevnou okrajovou podmínkou
- `force_nodes`: Množina ID uzlů s aplikovanou silou
- `output_file`: Cesta k výstupnímu souboru (bez přípony)

Pro vizualizaci v ParaView:
- 0: Bez okrajové podmínky
- 1: Pevná okrajová podmínka (fixed)
- 2: Aplikovaná síla (force)
"""
function export_boundary_conditions(grid::Grid, dh::DofHandler, 
                                   fixed_nodes::Set{Int}, force_nodes::Set{Int}, 
                                   output_file::String)
    println("Exportuji síť s okrajovými podmínkami do $output_file...")
    
    # Inicializace pole pro data uzlů (0 = bez BC, 1 = fixed, 2 = force)
    n_nodes = getnnodes(grid)
    bc_data = zeros(Int, n_nodes)
    
    # Označení uzlů s okrajovými podmínkami
    for node_id in fixed_nodes
        bc_data[node_id] = 1  # Pro pevné okrajové podmínky
    end
    
    for node_id in force_nodes
        bc_data[node_id] = 2  # Pro síly
    end
    
    # Vytvoření VTK souboru
    vtk_file = VTKGridFile(output_file, dh.grid) do vtk
        # Přidání dat o okrajových podmínkách jako pole na uzlech
        write_point_data(vtk, bc_data, "boundary_conditions")
    end
    
    println("Okrajové podmínky byly úspěšně exportovány do $(output_file).vtu")
end

end # module
