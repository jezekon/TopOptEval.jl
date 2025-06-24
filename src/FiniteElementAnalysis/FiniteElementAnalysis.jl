module FiniteElementAnalysis

using Ferrite
using LinearAlgebra
using SparseArrays
using StaticArrays

# Exported functions
export create_material_model, setup_problem, assemble_stiffness_matrix!,
       select_nodes_by_plane, select_nodes_by_circle, get_node_dofs,
       apply_fixed_boundary!, apply_sliding_boundary!, apply_force!, solve_system,
       calculate_stresses, create_simp_material_model, assemble_stiffness_matrix_simp!,
       calculate_stresses_simp, solve_system_simp

include("VolumeForce.jl")
export apply_volume_force!, apply_gravity!, apply_acceleration!, apply_variable_density_volume_force!

include("SimpleSolverConfig.jl")
export SimpleSolverConfig, SimpleSolverType, DIRECT, ITERATIVE,
       direct_solver, iterative_solver, auto_solve, estimate_memory_usage

"""
    create_material_model(youngs_modulus::Float64, poissons_ratio::Float64)

Creates material constants for a linearly elastic material.

Parameters:
- `youngs_modulus`: Young's modulus in Pa
- `poissons_ratio`: Poisson's ratio

Returns:
- lambda and mu coefficients for Hooke's law
"""
function create_material_model(youngs_modulus::Float64, poissons_ratio::Float64)
    # Lamé coefficients
    λ = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    μ = youngs_modulus / (2 * (1 + poissons_ratio))
    
    return λ, μ
end

"""
    constitutive_relation(ε, λ, μ)

Applies linear elastic relationship between strain and stress (Hooke's law).

Parameters:
- `ε`: strain tensor
- `λ`: first Lamé coefficient
- `μ`: second Lamé coefficient (shear modulus)

Returns:
- stress tensor
"""
function constitutive_relation(ε, λ, μ)
    # Linear elasticity: σ = λ*tr(ε)*I + 2μ*ε
    return λ * tr(ε) * one(ε) + 2μ * ε
end

"""
    setup_problem(grid::Grid, interpolation_order::Int=1)

Sets up the finite element problem for the given grid.
Automatically detects the cell type (tetrahedron or hexahedron) and creates appropriate interpolation.

Parameters:
- `grid`: Computational mesh (Ferrite Grid)
- `interpolation_order`: Order of the Lagrange interpolation (default: 1)

Returns:
- Tuple containing:
  - DofHandler for managing degrees of freedom
  - CellValues for interpolation and integration
  - Global stiffness matrix K
  - Global load vector f
"""
function setup_problem(grid::Grid, interpolation_order::Int=1)
    dim = 3  # 3D problem
    
    # Determine the cell type from the grid
    # We need to find what type of cells are in the grid
    # Get the first cell to determine the type
    cell_type = typeof(getcells(grid, 1))
    
    # Choose appropriate reference shape based on cell type
    if cell_type <: Ferrite.Hexahedron
        println("Setting up problem with hexahedral elements")
        ip = Lagrange{RefHexahedron, interpolation_order}()^dim  # vector interpolation for hexahedrons
        qr = QuadratureRule{RefHexahedron}(2)  # quadrature rule for hexahedrons
    else  # Default to tetrahedron
        println("Setting up problem with tetrahedral elements")
        ip = Lagrange{RefTetrahedron, interpolation_order}()^dim  # vector interpolation for tetrahedrons
        qr = QuadratureRule{RefTetrahedron}(2)  # quadrature rule for tetrahedrons
    end
    
    # Create cell values
    cellvalues = CellValues(qr, ip)
    
    # Set up the FE problem
    dh = DofHandler(grid)
    add!(dh, :u, ip)  # add displacement field
    close!(dh)
    
    # Allocate solution vectors and system matrices
    n_dofs = ndofs(dh)
    println("Number of DOFs: $n_dofs")
    K = allocate_matrix(dh)
    f = zeros(n_dofs)
    
    return dh, cellvalues, K, f
end


"""
    assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)

Assembles the global stiffness matrix and initializes the load vector.

Parameters:
- `K`: global stiffness matrix (modified in-place)
- `f`: global load vector (modified in-place)
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `λ`, `μ`: material parameters

Returns:
- nothing (modifies K and f in-place)
"""
function assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)
    # Element stiffness matrix and internal force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    
    # Create an assembler
    assembler = start_assemble(K, f)
    
    # Iterate over all cells and assemble global matrices
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        
        # Compute element stiffness matrix
        for q_point in 1:getnquadpoints(cellvalues)
            # Get integration weight
            dΩ = getdetJdV(cellvalues, q_point)
            
            for i in 1:n_basefuncs
                # Gradient of test function
                ∇Ni = shape_gradient(cellvalues, q_point, i)
                
                for j in 1:n_basefuncs
                    # Symmetric gradient of trial function
                    ∇Nj = shape_gradient(cellvalues, q_point, j)
                    
                    # Compute the small strain tensor
                    εi = symmetric(∇Ni)
                    εj = symmetric(∇Nj)
                    
                    # Apply constitutive law to get stress tensor
                    σ = constitutive_relation(εj, λ, μ)
                    
                    # Compute stiffness contribution using tensor double contraction
                    ke[i, j] += (εi ⊡ σ) * dΩ
                end
            end
        end
        
        # Assemble element contributions to global system
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    
    println("Stiffness matrix assembled successfully")
end

"""
    get_node_dofs(dh::DofHandler)

Creates a mapping from node IDs to their corresponding DOFs.

Parameters:
- `dh`: DofHandler

Returns:
- Dictionary mapping node IDs to their DOFs
"""
function get_node_dofs(dh::DofHandler)
    node_to_dofs = Dict{Int, Vector{Int}}()
    
    # For each cell, get the mapping of nodes to DOFs
    for cell in CellIterator(dh)
        cell_nodes = cell.nodes
        cell_dofs = celldofs(cell)
        
        # Assume that for each node we have 'dim' DOFs (one for each direction)
        # and they are arranged sequentially for each node
        nodes_per_cell = length(cell_nodes)
        dofs_per_node = length(cell_dofs) ÷ nodes_per_cell
        
        # For each node in the cell
        for (local_node_idx, global_node_idx) in enumerate(cell_nodes)
            # Calculate the range of DOFs for this node within the cell
            start_dof = (local_node_idx - 1) * dofs_per_node + 1
            end_dof = local_node_idx * dofs_per_node
            local_dofs = cell_dofs[start_dof:end_dof]
            
            # Add DOFs to the dictionary
            if !haskey(node_to_dofs, global_node_idx)
                node_to_dofs[global_node_idx] = local_dofs
            end
        end
    end
    
    return node_to_dofs
end

"""
    select_nodes_by_plane(grid::Grid, 
                          point::Vector{Float64}, 
                          normal::Vector{Float64}, 
                          tolerance::Float64=1e-6)

Selects nodes that lie on a plane defined by a point and normal vector.

Parameters:
- `grid`: Computational mesh
- `point`: A point on the plane [x, y, z]
- `normal`: Normal vector to the plane [nx, ny, nz]
- `tolerance`: Distance tolerance for node selection

Returns:
- Set of node IDs that lie on the plane
"""
function select_nodes_by_plane(grid::Grid, 
                               point::Vector{Float64}, 
                               normal::Vector{Float64}, 
                               tolerance::Float64=1e-4)
    # Normalize the normal vector
    unit_normal = normal / norm(normal)
    
    # Extract number of nodes
    num_nodes = getnnodes(grid)
    selected_nodes = Set{Int}()
    
    # Check each node
    for node_id in 1:num_nodes
        coord = grid.nodes[node_id].x
        
        # Calculate distance from point to plane: d = (p - p0) · n
        dist = abs(dot(coord - point, unit_normal))
        
        # If distance is within tolerance, node is on plane
        if dist < tolerance
            push!(selected_nodes, node_id)
        end
    end
    
    println("Selected $(length(selected_nodes)) nodes on the specified plane")
    return selected_nodes
end

"""
    select_nodes_by_circle(grid::Grid, 
                           center::Vector{Float64}, 
                           normal::Vector{Float64}, 
                           radius::Float64, 
                           tolerance::Float64=1e-6)

Selects nodes that lie on a circular region defined by center, normal and radius.

Parameters:
- `grid`: Computational mesh
- `center`: Center of the circle [x, y, z]
- `normal`: Normal vector to the plane containing the circle [nx, ny, nz]
- `radius`: Radius of the circle
- `tolerance`: Distance tolerance for node selection

Returns:
- Set of node IDs that lie on the circular region
"""
function select_nodes_by_circle(grid::Grid, 
                                center::Vector{Float64}, 
                                normal::Vector{Float64}, 
                                radius::Float64, 
                                tolerance::Float64=1e-6)
    # First, get nodes on the plane
    nodes_on_plane = select_nodes_by_plane(grid, center, normal, tolerance)
    
    # Normalize the normal vector
    unit_normal = normal / norm(normal)
    
    # Initialize set for nodes in circle
    nodes_in_circle = Set{Int}()
    
    # Check which nodes are within the circle radius
    for node_id in nodes_on_plane
        coord = grid.nodes[node_id].x
        
        # Project the vector from center to node onto the plane
        v = coord - center
        projection = v - dot(v, unit_normal) * unit_normal
        
        # Calculate distance from center in the plane
        dist = norm(projection)
        
        # If distance is less than radius, node is in the circle
        if dist <= radius + tolerance
            push!(nodes_in_circle, node_id)
        end
    end
    
    println("Selected $(length(nodes_in_circle)) nodes in the circular region")
    return nodes_in_circle
end

"""
    apply_fixed_boundary!(K, f, dh, nodes)

Applies fixed boundary conditions (all DOFs fixed) to the specified nodes.

Parameters:
- `K`: global stiffness matrix
- `f`: global load vector
- `dh`: DofHandler
- `nodes`: Set or Array of node IDs to be fixed

Returns:
- ConstraintHandler with the applied constraints
"""
function apply_fixed_boundary!(K, f, dh, nodes)
    dim = 3  # 3D problem
    
    # Create constraint handler
    ch = ConstraintHandler(dh)
    
    # Apply Dirichlet boundary conditions for fixed nodes
    # Fix each component individually
    for d in 1:dim
        dbc = Dirichlet(:u, nodes, (x, t) -> 0.0, d)
        add!(ch, dbc)
    end
    
    close!(ch)
    update!(ch, 0.0)
    apply!(K, f, ch)
    
    println("Applied fixed boundary conditions to $(length(nodes)) nodes")
    return ch
end

"""
    apply_sliding_boundary!(K, f, dh, nodes, fixed_dofs)

Applies sliding boundary conditions to the specified nodes,
allowing movement only in certain directions.

Parameters:
- `K`: global stiffness matrix
- `f`: global load vector
- `dh`: DofHandler
- `nodes`: Set or Array of node IDs for the sliding boundary
- `fixed_dofs`: Array of direction indices to fix (1=x, 2=y, 3=z)

Returns:
- ConstraintHandler with the applied constraints
"""
function apply_sliding_boundary!(K, f, dh, nodes, fixed_dofs)
    # Create constraint handler
    ch = ConstraintHandler(dh)
    
    # Apply Dirichlet boundary conditions only for specified directions
    for d in fixed_dofs
        dbc = Dirichlet(:u, nodes, (x, t) -> 0.0, d)
        add!(ch, dbc)
    end
    
    close!(ch)
    update!(ch, 0.0)
    apply!(K, f, ch)
    
    println("Applied sliding boundary conditions to $(length(nodes)) nodes, fixing DOFs: $fixed_dofs")
    return ch
end

"""
    apply_force!(f, dh, nodes, force_vector)

Applies a force to the specified nodes.

Parameters:
- `f`: global load vector (modified in-place)
- `dh`: DofHandler
- `nodes`: Array or Set of node IDs where force is applied
- `force_vector`: Force vector [Fx, Fy, Fz] in Newtons

Returns:
- nothing (modifies f in-place)
"""
function apply_force!(f, dh, nodes, force_vector)
    if isempty(nodes)
        error("No nodes provided for force application.")
    end
    
    # Get mapping from nodes to DOFs
    node_to_dofs = get_node_dofs(dh)
    
    # Calculate force per node
    force_per_node = force_vector ./ length(nodes)
    
    # Apply force to each node
    for node_id in nodes
        if haskey(node_to_dofs, node_id)
            dofs = node_to_dofs[node_id]
            
            # Apply force components to respective DOFs
            for (i, component) in enumerate(force_per_node)
                if i <= length(dofs)
                    f[dofs[i]] += component
                end
            end
        end
    end
    
    println("Applied force $force_vector distributed over $(length(nodes)) nodes")
end

"""
    calculate_stresses(u, dh, cellvalues, λ, μ)

Calculate stress field from displacement solution.

Parameters:
- `u`: displacement vector
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `λ`, `μ`: material parameters

Returns:
- Tuple with:
  - Dictionary mapping cell IDs to stress tensors at quadrature points
  - Maximum von Mises stress value in the model
  - Cell ID where the maximum stress occurs
"""
function calculate_stresses(u, dh, cellvalues, λ, μ)
    # Initialize storage for stresses
    # We'll store stresses at quadrature points for each cell
    stress_field = Dict{Int, Vector{SymmetricTensor{2, 3, Float64}}}()
    
    # Track maximum von Mises stress and its location
    max_von_mises = 0.0
    max_vm_cell_id = 0
    
    # For each cell, calculate stresses at quadrature points
    for cell in CellIterator(dh)
        cell_id = cellid(cell)
        cell_dofs = celldofs(cell)
        
        # Get displacements for this cell
        u_cell = u[cell_dofs]
        
        # Initialize cell values for this cell
        reinit!(cellvalues, cell)
        
        # Initialize storage for stress at each quadrature point
        n_qpoints = getnquadpoints(cellvalues)
        cell_stresses = Vector{SymmetricTensor{2, 3, Float64}}(undef, n_qpoints)
        
        # Calculate cell's average stress for von Mises calculation
        avg_stress = zero(SymmetricTensor{2, 3, Float64})
        
        # Compute stresses at each quadrature point
        for q_point in 1:n_qpoints
            # Calculate strain from displacement gradients
            # For a vector-valued field, function_gradient returns a tensor
            grad_u = function_gradient(cellvalues, q_point, u_cell)
            
            # Calculate small strain tensor
            ε = symmetric(grad_u)
            
            # Calculate stress using constitutive relation
            σ = constitutive_relation(ε, λ, μ)
            
            # Store stress for this quadrature point
            cell_stresses[q_point] = σ
            
            # Add to average stress
            avg_stress += σ
        end
        
        # Compute average stress for the cell
        if n_qpoints > 0
            avg_stress = avg_stress / n_qpoints
            
            # Calculate von Mises stress for this cell
            # von Mises = sqrt(3/2 * (dev(σ) ⊡ dev(σ)))
            cell_von_mises = sqrt(3/2 * (dev(avg_stress) ⊡ dev(avg_stress)))
            
            # Update maximum if this is higher
            if cell_von_mises > max_von_mises
                max_von_mises = cell_von_mises
                max_vm_cell_id = cell_id
            end
        end
        
        # Store stresses for this cell
        stress_field[cell_id] = cell_stresses
    end
    
    println("Stress calculation complete")
    println("Maximum von Mises stress: $max_von_mises at cell $max_vm_cell_id")
    
    return stress_field, max_von_mises, max_vm_cell_id
end

"""
    solve_system(K, f, dh, cellvalues, λ, μ, constraints...; 
                solver_config::Union{SimpleSolverConfig, Nothing} = nothing)

Vylepšená solve_system funkce s volitelnou konfigurací solveru.

# Parameters:
- `K`: globální matice tuhosti
- `f`: globální vektor zatížení
- `dh`: DofHandler
- `cellvalues`: CellValues pro interpolaci a integraci
- `λ`, `μ`: materiálové parametry
- `constraints...`: ConstraintHandlers s okrajovými podmínkami
- `solver_config`: Volitelná konfigurace solveru (Nothing = automatický výběr)

# Returns:
- Tuple containing:
  - displacement vector
  - deformation energy  
  - stress field dictionary
  - maximum von Mises stress value
  - cell ID where the maximum stress occurs
  - solver information dictionary
"""
function solve_system(K, f, dh, cellvalues, λ, μ, constraints...; 
                     solver_config::Union{SimpleSolverConfig, Nothing} = nothing)
    
    # Aplikuj okrajové podmínky
    for ch in constraints
        apply_zero!(K, f, ch)
    end
    
    println("Příprava řešení lineárního systému...")
    
    # Pokud není zadána konfigurace, použij automatický výběr
    if solver_config === nothing
        println("Automatický výběr solveru...")
        x, solver_info = auto_solve(K, f, verbose=true)
    else
        println("Používám zadanou konfiguraci solveru...")
        x, solver_info = solve_linear_system_simple(K, f, solver_config)
    end
    
    # Kontrola konvergence pro iterativní solvery
    if !solver_info["converged"] && haskey(solver_info, "iterations")
        @warn "Iterativní solver nekonvergoval!"
        @warn "Finální reziduum: $(solver_info["residual_norm"])"
        @warn "Zvažte:"
        @warn "  - Zvýšení max_iterations"
        @warn "  - Snížení tolerance"
        @warn "  - Použití přímého solveru: direct_solver()"
    end
    
    # Výpočet deformační energie: U = 0.5 * u^T * K * u
    println("Výpočet deformační energie...")
    deformation_energy = 0.5 * dot(x, K * x)
    
    # Výpočet napětí
    println("Výpočet pole napětí...")
    stress_field, max_von_mises, max_stress_cell = calculate_stresses(x, dh, cellvalues, λ, μ)
    
    # Výsledky
    println("\n" * "="^60)
    println("ANALÝZA DOKONČENA")
    println("="^60)
    println("Solver: $(solver_info["method"])")
    if haskey(solver_info, "iterations")
        println("Iterace: $(solver_info["iterations"])")
        println("Finální reziduum: $(solver_info["residual_norm"])")
    end
    println("Čas řešení: $(round(solver_info["solve_time"], digits=2)) sekund")
    println("Deformační energie: $deformation_energy J")
    println("Maximální von Mises napětí: $max_von_mises v elementu $max_stress_cell")
    
    return x, deformation_energy, stress_field, max_von_mises, max_stress_cell, solver_info
end

"""
    create_simp_material_model(E0::Float64, nu::Float64, Emin::Float64=1e-9, p::Float64=3.0)

Creates a material model using the SIMP (Solid Isotropic Material with Penalization) approach.

Parameters:
- `E0`: Base material Young's modulus
- `nu`: Poisson's ratio
- `Emin`: Minimum Young's modulus (default: 1e-9)
- `p`: Penalization power (default: 3.0)

Returns:
- Function mapping density to Lamé parameters (λ, μ)
"""
function create_simp_material_model(E0::Float64, nu::Float64, Emin::Float64=1e-6, p::Float64=3.0)
    function material_for_density(density::Float64)
        # SIMP model: E(ρ) = Emin + (E0 - Emin) * ρ^p
        E = Emin + (E0 - Emin) * density^p
        
        # Calculate Lamé parameters
        λ = E * nu / ((1 + nu) * (1 - 2 * nu))
        μ = E / (2 * (1 + nu))
        
        return λ, μ
    end
    
    return material_for_density
end

"""
    assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, density_data)

Assembles the global stiffness matrix using variable material properties based on element density.

Parameters:
- `K`: global stiffness matrix (modified in-place)
- `f`: global load vector (modified in-place)
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `material_model`: Function mapping density to material parameters (λ, μ)
- `density_data`: Vector with density values for each cell

Returns:
- nothing (modifies K and f in-place)
"""
function assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, density_data)
    # Element stiffness matrix and internal force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    
    # Create an assembler
    assembler = start_assemble(K, f)
    
    # Iterate over all cells and assemble global matrices
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        
        # Get cell ID and density
        cell_id = cellid(cell)
        density = density_data[cell_id]
        
        # Get material parameters for this density
        λ, μ = material_model(density)
        
        # Compute element stiffness matrix
        for q_point in 1:getnquadpoints(cellvalues)
            # Get integration weight
            dΩ = getdetJdV(cellvalues, q_point)
            
            for i in 1:n_basefuncs
                # Gradient of test function
                ∇Ni = shape_gradient(cellvalues, q_point, i)
                
                for j in 1:n_basefuncs
                    # Symmetric gradient of trial function
                    ∇Nj = shape_gradient(cellvalues, q_point, j)
                    
                    # Compute the small strain tensor
                    εi = symmetric(∇Ni)
                    εj = symmetric(∇Nj)
                    
                    # Apply constitutive law to get stress tensor
                    σ = constitutive_relation(εj, λ, μ)
                    
                    # Compute stiffness contribution using tensor double contraction
                    ke[i, j] += (εi ⊡ σ) * dΩ
                end
            end
        end
        
        # Assemble element contributions to global system
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    
    println("Stiffness matrix assembled successfully with variable material properties")
end

"""
    calculate_stresses_simp(u, dh, cellvalues, material_model, density_data)

Calculate stress field from displacement solution, using variable material properties.

Parameters:
- `u`: displacement vector
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `material_model`: Function mapping density to material parameters (λ, μ)
- `density_data`: Vector with density values for each cell

Returns:
- Tuple with:
  - Dictionary mapping cell IDs to stress tensors at quadrature points
  - Maximum von Mises stress value in the model
  - Cell ID where the maximum stress occurs
"""
function calculate_stresses_simp(u, dh, cellvalues, material_model, density_data)
    # Initialize storage for stresses
    stress_field = Dict{Int, Vector{SymmetricTensor{2, 3, Float64}}}()
    
    # Track maximum von Mises stress and its location
    max_von_mises = 0.0
    max_vm_cell_id = 0
    
    # For each cell, calculate stresses at quadrature points
    for cell in CellIterator(dh)
        cell_id = cellid(cell)
        cell_dofs = celldofs(cell)
        
        # Get displacements for this cell
        u_cell = u[cell_dofs]
        
        # Get density and material properties for this cell
        density = density_data[cell_id]
        λ, μ = material_model(density)
        
        # Initialize cell values for this cell
        reinit!(cellvalues, cell)
        
        # Initialize storage for stress at each quadrature point
        n_qpoints = getnquadpoints(cellvalues)
        cell_stresses = Vector{SymmetricTensor{2, 3, Float64}}(undef, n_qpoints)
        
        # Calculate cell's average stress for von Mises calculation
        avg_stress = zero(SymmetricTensor{2, 3, Float64})
        
        # Compute stresses at each quadrature point
        for q_point in 1:n_qpoints
            # Calculate strain from displacement gradients
            grad_u = function_gradient(cellvalues, q_point, u_cell)
            
            # Calculate small strain tensor
            ε = symmetric(grad_u)
            
            # Calculate stress using constitutive relation
            σ = constitutive_relation(ε, λ, μ)
            
            # Store stress for this quadrature point
            cell_stresses[q_point] = σ
            
            # Add to average stress
            avg_stress += σ
        end
        
        # Compute average stress for the cell
        if n_qpoints > 0
            avg_stress = avg_stress / n_qpoints
            
            # Calculate von Mises stress for this cell
            # von Mises = sqrt(3/2 * (dev(σ) ⊡ dev(σ)))
            cell_von_mises = sqrt(3/2 * (dev(avg_stress) ⊡ dev(avg_stress)))
            
            # Update maximum if this is higher
            if cell_von_mises > max_von_mises
                max_von_mises = cell_von_mises
                max_vm_cell_id = cell_id
            end
        end
        
        # Store stresses for this cell
        stress_field[cell_id] = cell_stresses
    end
    
    println("Stress calculation complete with variable material properties")
    println("Maximum von Mises stress: $max_von_mises at cell $max_vm_cell_id")
    
    return stress_field, max_von_mises, max_vm_cell_id
end

"""
    solve_system_simp(K, f, dh, cellvalues, material_model, density_data, constraints...;
                     solver_config::Union{SimpleSolverConfig, Nothing} = nothing)

Vylepšená solve_system_simp funkce pro SIMP topologickou optimalizaci.

# Parameters:
- `K`: globální matice tuhosti
- `f`: globální vektor zatížení
- `dh`: DofHandler
- `cellvalues`: CellValues pro interpolaci a integraci
- `material_model`: Funkce mapující hustotu na materiálové parametry (λ, μ)
- `density_data`: Vektor s hodnotami hustoty pro každý element
- `constraints...`: ConstraintHandlers s okrajovými podmínkami
- `solver_config`: Volitelná konfigurace solveru

# Returns:
- Tuple containing:
  - displacement vector
  - deformation energy
  - stress field dictionary
  - maximum von Mises stress value
  - cell ID where the maximum stress occurs
  - solver information dictionary
"""
function solve_system_simp(K, f, dh, cellvalues, material_model, density_data, constraints...;
                          solver_config::Union{SimpleSolverConfig, Nothing} = nothing)
    
    # Aplikuj okrajové podmínky
    for ch in constraints
        apply_zero!(K, f, ch)
    end
    
    println("Příprava řešení SIMP lineárního systému...")
    
    # Automatický výběr solveru s ohledem na SIMP specifika
    if solver_config === nothing
        println("Automatický výběr solveru pro SIMP analýzu...")
        
        # Pro SIMP problémy bývá matice hůře podmíněná kvůli nízkým hustotám
        matrix_size = size(K, 1)
        nnz_count = nnz(K)
        total_ram_gb = Sys.total_memory() / 1e9
        available_ram_gb = total_ram_gb * 0.6  # Konzervativnější pro SIMP
        
        # Kontrola minimální hustoty - pokud jsou velmi nízké hustoty, může být problém s konvergencí
        min_density = minimum(density_data)
        if min_density < 1e-6
            println("  Detekována velmi nízká hustota ($(min_density))")
            println("  Matice může být špatně podmíněná")
        end
        
        config = choose_solver_automatically(matrix_size, nnz_count, available_ram_gb)
        
        # Pro SIMP s nízkými hustotami zvyš toleranci
        if min_density < 1e-3 && config.solver_type == ITERATIVE
            config = SimpleSolverConfig(ITERATIVE, 
                                      max_iterations=config.max_iterations,
                                      tolerance=1e-5,  # Mírnější tolerance
                                      verbose=true)
            println("  Upravena tolerance pro SIMP: 1e-5")
        end
        
        x, solver_info = solve_linear_system_simple(K, f, config)
    else
        println("Používám zadanou konfiguraci pro SIMP...")
        x, solver_info = solve_linear_system_simple(K, f, solver_config)
    end
    
    # Kontrola konvergence
    if !solver_info["converged"] && haskey(solver_info, "iterations")
        @warn "SIMP solver nekonvergoval!"
        @warn "Možné příčiny:"
        @warn "  - Velmi nízké hustoty způsobují špatné podmínění matice"
        @warn "  - Zvyšte minimální hustotu (Emin) v material_model"
        @warn "  - Použijte přímý solver: direct_solver()"
        @warn "  - Snižte toleranci: iterative_solver(tolerance=1e-4)"
    end
    
    # Výpočet deformační energie
    println("Výpočet deformační energie...")
    deformation_energy = 0.5 * dot(x, K * x)
    
    # Výpočet napětí s proměnnými materiálovými vlastnostmi
    println("Výpočet pole napětí pro SIMP...")
    stress_field, max_von_mises, max_stress_cell = calculate_stresses_simp(x, dh, cellvalues, material_model, density_data)
    
    # Výsledky
    println("\n" * "="^60)
    println("SIMP ANALÝZA DOKONČENA")
    println("="^60)
    println("Solver: $(solver_info["method"])")
    if haskey(solver_info, "iterations")
        println("Iterace: $(solver_info["iterations"])")
        println("Finální reziduum: $(solver_info["residual_norm"])")
    end
    println("Čas řešení: $(round(solver_info["solve_time"], digits=2)) sekund")
    println("Deformační energie: $deformation_energy J")
    println("Maximální von Mises napětí: $max_von_mises v elementu $max_stress_cell")
    
    return x, deformation_energy, stress_field, max_von_mises, max_stress_cell, solver_info
end

"""
    analyze_problem_memory(dh::DofHandler; available_ram_gb::Float64 = 0.0)

Analyzuje velikost problému a doporučí vhodný solver z hlediska paměti.

# Parameters:
- `dh`: DofHandler z FEM problému
- `available_ram_gb`: Dostupná RAM v GB (0 = auto-detekce)

# Returns:
- Dictionary se statistikami a doporučením
"""
function analyze_problem_memory(dh::DofHandler; available_ram_gb::Float64 = 0.0)
    n_dofs = ndofs(dh)
    n_nodes = numnodes(dh.grid)
    n_elements = numcells(dh.grid)
    
    # Odhad počtu nenulových prvků pro 3D FEM
    avg_connections = 27  # Konzervativní odhad pro strukturovanou 3D síť
    estimated_nnz = min(n_dofs * avg_connections, n_dofs^2)
    
    # Auto-detekce RAM
    if available_ram_gb <= 0.0
        total_ram_gb = Sys.total_memory() / 1e9
        available_ram_gb = total_ram_gb * 0.7
    end
    
    # Odhady paměti
    direct_memory = estimate_memory_usage(n_dofs, estimated_nnz, DIRECT)
    iterative_memory = estimate_memory_usage(n_dofs, estimated_nnz, ITERATIVE)
    
    println("\n" * "="^60)
    println("ANALÝZA VELIKOSTI PROBLÉMU")
    println("="^60)
    println("Uzly: $(n_nodes)")
    println("Elementy: $(n_elements)")
    println("Stupně volnosti: $(n_dofs)")
    println("Odhadované nenulové prvky: $(estimated_nnz)")
    println("Hustota matice: $(round(100 * estimated_nnz / n_dofs^2, digits=3))%")
    println("\nOdhady paměti:")
    println("  Přímý solver: $(round(direct_memory, digits=2)) GB")
    println("  Iterativní solver: $(round(iterative_memory, digits=2)) GB")
    println("  Dostupná RAM: $(round(available_ram_gb, digits=1)) GB")
    
    # Doporučení
    println("\nDoporučení:")
    if direct_memory < available_ram_gb * 0.7
        println("  ✓ Přímý solver by měl fungovat dobře")
        recommendation = direct_solver()
    elseif iterative_memory < available_ram_gb * 0.9
        println("  → Doporučuji iterativní solver pro úsporu paměti")
        max_iter = max(1000, n_dofs ÷ 100)
        recommendation = iterative_solver(max_iterations=max_iter, tolerance=1e-8, verbose=true)
    else
        println("  ⚠ Problém může být příliš velký pro dostupnou RAM")
        println("    Zvažte:")
        println("    - Hrubší síť")
        println("    - Více RAM")
        println("    - Distribuované výpočty")
        recommendation = iterative_solver(max_iterations=2000, tolerance=1e-5, verbose=true)
    end
    
    return Dict(
        "n_dofs" => n_dofs,
        "n_nodes" => n_nodes,
        "n_elements" => n_elements,
        "estimated_nnz" => estimated_nnz,
        "direct_memory_gb" => direct_memory,
        "iterative_memory_gb" => iterative_memory,
        "available_ram_gb" => available_ram_gb,
        "recommendation" => recommendation
    )
end

end # module
