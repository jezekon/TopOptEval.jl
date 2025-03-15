# Load necessary packages
using Ferrite
using LinearAlgebra
using SparseArrays
using WriteVTK
using StaticArrays
using FerriteGmsh  # Required for parsing GMSH (.msh) files

"""
    analyze_cantilever_beam(
        grid::Grid,
        youngs_modulus::Float64=210.0e9, 
        poissons_ratio::Float64=0.3
    )

Perform finite element analysis on a cantilever beam discretized with tetrahedral elements.
Calculate the deformation energy when a 1N force is applied.

Parameters:
- `grid`: Ferrite Grid containing the mesh
- `youngs_modulus`: Young's modulus of the material in Pa (default: 210 GPa, steel)
- `poissons_ratio`: Poisson's ratio of the material (default: 0.3, steel)

Returns:
- Tuple containing (displacement_vector, deformation_energy, dof_handler)
"""
function analyze_cantilever_beam(
    grid::Grid,
    youngs_modulus::Float64=210.0e9, 
    poissons_ratio::Float64=0.3
)
    # Extract problem dimensions
    num_nodes = getnnodes(grid)
    num_cells = getncells(grid)
    
    println("Problem size: $num_nodes nodes, $num_cells tetrahedral elements")
    
    # Create material model (linear elastic)
    # Using a material that respects Hooke's law for linear elasticity
    λ = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    μ = youngs_modulus / (2 * (1 + poissons_ratio))
    
    # Create constitutive relation for linear elasticity
    function constitutive_relation(ε)
        # Linear elasticity: σ = λ*tr(ε)*I + 2μ*ε
        return λ * tr(ε) * one(ε) + 2μ * ε
    end
    
    # Create the finite element space
    # Use linear shape functions for displacement
    dim = 3  # 3D problem
    ip = Lagrange{dim, RefTetrahedron, 1}()
    qr = QuadratureRule{dim, RefTetrahedron}(2)
    cellvalues = CellVectorValues(qr, ip)
    
    # Set up the FE problem
    dh = DofHandler(grid)
    add!(dh, :u, dim)  # 3 displacement components per node (3D problem)
    close!(dh)
    
    # Allocate solution vectors and system matrices
    n_dofs = ndofs(dh)
    K = create_sparsity_pattern(dh)
    f = zeros(n_dofs)
    
    # Assemble stiffness matrix
    assembler = start_assemble(K, f)
    
    # Element stiffness matrix and internal force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(dim * n_basefuncs, dim * n_basefuncs)
    fe = zeros(dim * n_basefuncs)
    
    # Iterate over all cells and assemble global matrices
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        
        # Compute element stiffness matrix
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            
            for i in 1:n_basefuncs
                ∇Ni = shape_gradient(cellvalues, q_point, i)
                
                for j in 1:n_basefuncs
                    ∇Nj = shape_gradient(cellvalues, q_point, j)
                    
                    # Compute the small strain tensor
                    εi = symmetric(∇Ni)
                    εj = symmetric(∇Nj)
                    
                    # Apply constitutive law to get stress tensor
                    σ = constitutive_relation(εj)
                    
                    # Compute the inner product of strain and stress tensors
                    # for stiffness contribution
                    for d1 in 1:dim
                        for d2 in 1:dim
                            ke[(i-1)*dim + d1, (j-1)*dim + d2] += εi[d1, d2] * σ[d1, d2] * dΩ
                        end
                    end
                end
            end
        end
        
        # Assemble element contributions to global system
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    
    # Apply boundary conditions
    
    # 1. Fixed nodes at x=0 (with tolerance 10^-6)
    tol = 1e-6  # Tolerance as specified for floating point comparison
    fixed_nodes = Int[]
    
    # Find nodes on yz plane (x ≈ 0)
    for node_id in 1:num_nodes
        coord = node_coordinates(grid, node_id)
        if abs(coord[1]) < tol
            push!(fixed_nodes, node_id)
        end
    end
    
    println("Number of fixed nodes: $(length(fixed_nodes))")
    
    # Create and apply Dirichlet boundary conditions for fixed nodes (all DOFs constrained)
    ch = ConstraintHandler(dh)
    dbc = Dirichlet(:u, Set(fixed_nodes), (x, t) -> zeros(Vec{3, Float64}), collect(1:dim))
    add!(ch, dbc)
    close!(ch)
    apply!(K, f, ch)
    
    # 2. Force of 1N applied at x=60 (with tolerance 10^-6)
    load_nodes = Int[]
    
    # Find nodes on yz plane where x ≈ 60
    for node_id in 1:num_nodes
        coord = node_coordinates(grid, node_id)
        if abs(coord[1] - 60.0) < tol
            push!(load_nodes, node_id)
        end
    end
    
    println("Number of load nodes: $(length(load_nodes))")
    
    if isempty(load_nodes)
        error("No load nodes found at x = 60.0 ± $tol. Check your mesh geometry.")
    end
    
    # Apply total force of 1N, distributed evenly among the load nodes
    # Force in +x direction
    force_per_node = [1.0, 0.0, 0.0] / length(load_nodes)
    
    # Add nodal forces to global force vector
    for node_id in load_nodes
        # Get global dofs for this node
        node_dofs = zeros(Int, dim)
        for d in 1:dim
            node_dofs[d] = dh.cell_dofs[celldof(dh, :u, d, node_id)]
        end
        
        # Apply force components to appropriate DOFs
        for d in 1:dim
            f[node_dofs[d]] += force_per_node[d]
        end
    end
    
    # Solve the system
    println("Solving linear system with $(n_dofs) degrees of freedom...")
    # Apply boundary conditions
    apply_zero!(K, f, ch)
    
    # Solve
    u = zeros(n_dofs)
    solver = Ferrite.JuliaSolver()  # Using built-in linear solver
    solve!(u, solver, K, f)
    
    # Calculate deformation energy: U = 0.5 * u^T * K * u
    deformation_energy = 0.5 * dot(u, K * u)
    
    println("Analysis complete.")
    println("Deformation energy: $deformation_energy J")
    
    return u, deformation_energy, dh
end

"""
    import_and_analyze_mesh(
        mesh_file::String, 
        youngs_modulus::Float64=210.0e9, 
        poissons_ratio::Float64=0.3
    )

Import a mesh file and analyze the cantilever beam.
Currently supports only .msh files from GMSH.

Parameters:
- `mesh_file`: Path to GMSH .msh file
- `youngs_modulus`: Young's modulus of the material in Pa (default: 210 GPa, steel)
- `poissons_ratio`: Poisson's ratio of the material (default: 0.3, steel)

Returns:
- Tuple containing (displacement_vector, deformation_energy, dof_handler)
"""
function import_and_analyze_mesh(mesh_file::String, youngs_modulus::Float64=210.0e9, poissons_ratio::Float64=0.3)
    # Check file extension
    ext = lowercase(splitext(mesh_file)[2])
    
    if ext != ".msh"
        error("Unsupported mesh format: $ext. Only .msh format is supported.")
    end
    
    println("Importing mesh from $mesh_file...")
    
    # Import the GMSH .msh file using FerriteGmsh
    # FerriteGmsh.togrid is the correct function to convert GMSH files to Ferrite.Grid
    grid = FerriteGmsh.togrid(mesh_file)
    
    # Analyze the mesh
    return analyze_cantilever_beam(grid, youngs_modulus, poissons_ratio)
end

"""
    export_results(
        displacements::Vector{Float64}, 
        dh::DofHandler, 
        output_file::String
    )

Export the FEM results to a VTK file for visualization.

Parameters:
- `displacements`: Solution vector containing displacements
- `dh`: DofHandler used in the analysis
- `output_file`: Output file path
"""
function export_results(displacements::Vector{Float64}, dh::DofHandler, output_file::String)
    println("Exporting results to $output_file...")
    
    # Create a VTK file to visualize the results
    vtk_grid = vtk_grid(output_file, dh)
    
    # Add displacement field to the VTK file
    vtk_point_data(vtk_grid, dh, displacements, :displacement)
    
    # Write the results
    vtk_save(vtk_grid)
    println("Results exported successfully.")
end

# Main execution code
function main()
    println("FEM Analysis of Cantilever Beam with Tetrahedral Elements")
    println("=========================================================")
    
    # Mesh file to analyze
    mesh_file = "data/cantilever_beam_volume_mesh.msh"
    println("Processing $mesh_file...")
    
    # Run the analysis
    displacements, energy, dh = import_and_analyze_mesh(mesh_file)
    
    # Export results for visualization
    export_results(displacements, dh, "cantilever_beam_results")
    
    println("Final deformation energy: $energy J")
    
    return displacements, energy
end

    main()
