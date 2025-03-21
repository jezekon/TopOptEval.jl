module TopOptEval

# Include submodules
include("MeshImport/MeshImport.jl")
include("FiniteElementAnalysis/FiniteElementAnalysis.jl")
include("ResultsExport/ResultsExport.jl")
include("Utils/Utils.jl")

# Export submodules for direct access
export MeshImport, FiniteElementAnalysis, ResultsExport

# Re-export commonly used functions for convenience
export import_mesh, create_material_model, setup_problem, 
       assemble_stiffness_matrix!, select_nodes_by_plane, 
       select_nodes_by_circle, apply_fixed_boundary!,
       apply_sliding_boundary!, apply_force!, solve_system, 
       export_results, get_node_dofs, export_boundary_conditions,
       # Add these SIMP-related exports
       create_simp_material_model, assemble_stiffness_matrix_simp!,
       solve_system_simp, extract_cell_density

# Import specific functions from submodules to re-export
import .MeshImport: import_mesh, extract_cell_density
import .FiniteElementAnalysis: create_material_model, setup_problem,
       assemble_stiffness_matrix!, select_nodes_by_plane,
       select_nodes_by_circle, apply_fixed_boundary!,
       apply_sliding_boundary!, apply_force!, solve_system,
       get_node_dofs, create_simp_material_model, 
       assemble_stiffness_matrix_simp!, solve_system_simp
import .ResultsExport: export_results, export_boundary_conditions
import .Utils: calculate_volume

end # module TopOptEval

