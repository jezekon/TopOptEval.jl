module TopOptEval

# Include submodules
include("MeshImport/MeshImport.jl")
using .MeshImport

include("FiniteElementAnalysis/FiniteElementAnalysis.jl")
using .FiniteElementAnalysis

include("ResultsExport/ResultsExport.jl")
using .ResultsExport

include("Utils/Utils.jl")
using .Utils

# Re-export commonly used functions for convenience
export import_mesh, create_material_model, setup_problem, 
       assemble_stiffness_matrix!, select_nodes_by_plane, 
       select_nodes_by_circle, apply_fixed_boundary!,
       apply_sliding_boundary!, apply_force!, solve_system, 
       export_results, get_node_dofs, export_boundary_conditions,
       # Add these SIMP-related exports
       create_simp_material_model, assemble_stiffness_matrix_simp!,
       solve_system_simp, extract_cell_density

# Add volume force exports
export apply_volume_force!, apply_gravity!, apply_acceleration!, apply_variable_density_volume_force!

# Add solver configuration exports (MISSING - this was the problem!)
export SimpleSolverConfig, SimpleSolverType, DIRECT, ITERATIVE,
       direct_solver, iterative_solver, auto_solve, estimate_memory_usage

end # module TopOptEval
