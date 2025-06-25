"""
Enhanced solver module for large-scale FEM problems with Krylov methods
and memory-efficient strategies.

This module provides robust solving capabilities for FEM systems using
modern Krylov subspace methods with various preconditioning options.
"""

using LinearAlgebra
using SparseArrays
using Preconditioners
using Printf

# Try to load Krylov.jl (preferred) or IterativeSolvers.jl (fallback)
const has_krylov = try
    using Krylov
    true
catch
    false
end

const has_iterativesolvers = try
    using IterativeSolvers
    true
catch
    false
end

if !has_krylov && !has_iterativesolvers
    error("Either Krylov.jl or IterativeSolvers.jl must be installed")
end

# Prefer Krylov.jl if available
const USE_KRYLOV = has_krylov

# Solver configuration structure
struct SolverConfig
    method::Symbol              # :direct, :cg, :minres, :gmres, :bicgstab, :auto
    preconditioner::Symbol      # :none, :diagonal, :ilu, :ichol
    tolerance::Float64          # Convergence tolerance
    max_iterations::Int         # Maximum iterations
    memory_limit::Float64       # Memory limit in GB
    verbose::Bool               # Print solver information
    restart::Int                # GMRES restart parameter
    drop_tolerance::Float64     # ILU/IChol drop tolerance
    history::Bool               # Store convergence history
end

# Default configuration
function SolverConfig(;
    method::Symbol = :auto,
    preconditioner::Symbol = :diagonal,
    tolerance::Float64 = 1e-8,
    max_iterations::Int = 0,  # 0 means automatic
    memory_limit::Float64 = Sys.total_memory() / 1e9 * 0.8,
    verbose::Bool = true,
    restart::Int = 30,
    drop_tolerance::Float64 = 1e-4,
    history::Bool = false
)
    # Set default max_iterations based on expected problem size
    if max_iterations == 0
        max_iterations = 10000  # Conservative default
    end
    
    return SolverConfig(method, preconditioner, tolerance, max_iterations, 
                       memory_limit, verbose, restart, drop_tolerance, history)
end

"""
    estimate_memory_usage(K::SparseMatrixCSC)

Estimates memory usage for different solver methods.
Returns a Dict with memory estimates in GB.
"""
function estimate_memory_usage(K::SparseMatrixCSC)
    n = size(K, 1)
    nnz_K = nnz(K)
    
    # Základní paměť matice
    matrix_memory = (nnz_K * 8 + n * 8) / 1e9
    vector_memory = n * 8 / 1e9
    
    # Realistický fill-in pro velké FEM matice
    # Místo optimistických 10%, použijeme konzervativní odhad
    if n > 500000
        fill_factor = min(50.0, n / 20000)  # 50x-100x pro velké matice
    elseif n > 100000
        fill_factor = min(20.0, n / 10000)  # 20x-50x pro střední matice
    else
        fill_factor = 5.0  # 5x pro malé matice
    end
    
    # Odhady paměti
    direct_memory = matrix_memory * (1 + fill_factor)
    cg_memory = matrix_memory + 6 * vector_memory
    gmres_memory = matrix_memory + 35 * vector_memory
    
    return Dict(
        :direct => direct_memory,
        :cg => cg_memory,
        :gmres => gmres_memory,
        :matrix_only => matrix_memory
    )
end

"""
    estimate_bandwidth(K::SparseMatrixCSC)

Estimates the bandwidth of a sparse matrix.
"""
function estimate_bandwidth(K::SparseMatrixCSC)
    rows = rowvals(K)
    n = size(K, 1)
    max_bandwidth = 0
    
    for j in 1:n
        col_range = nzrange(K, j)
        if !isempty(col_range)
            col_rows = rows[col_range]
            if !isempty(col_rows)
                max_bandwidth = max(max_bandwidth, maximum(abs.(col_rows .- j)))
            end
        end
    end
    
    return max_bandwidth
end

"""
    check_matrix_properties(K::SparseMatrixCSC)

Analyzes matrix properties to guide solver selection.
"""
function check_matrix_properties(K::SparseMatrixCSC)
    n = size(K, 1)
    
    # Check symmetry (efficiently for sparse matrices)
    is_symmetric = true
    rows = rowvals(K)
    vals = nonzeros(K)
    
    # Sample check for symmetry (full check is expensive)
    sample_size = min(100, n ÷ 10)
    for j in 1:sample_size:n
        for idx in nzrange(K, j)
            i = rows[idx]
            if i != j  # Skip diagonal
                # Check if K[j,i] exists and equals K[i,j]
                found = false
                for idx2 in nzrange(K, i)
                    if rows[idx2] == j
                        if abs(vals[idx] - vals[idx2]) > 1e-10
                            is_symmetric = false
                            break
                        end
                        found = true
                        break
                    end
                end
                if !found && abs(vals[idx]) > 1e-10
                    is_symmetric = false
                    break
                end
            end
        end
        if !is_symmetric
            break
        end
    end
    
    # Check positive definiteness (heuristic)
    # For FEM matrices, usually positive definite after BC application
    is_positive_definite = true
    for j in 1:n
        diag_val = 0.0
        for idx in nzrange(K, j)
            if rows[idx] == j
                diag_val = vals[idx]
                break
            end
        end
        if diag_val <= 0
            is_positive_definite = false
            break
        end
    end
    
    return (symmetric = is_symmetric, positive_definite = is_positive_definite)
end

"""
    select_solver_method(K::SparseMatrixCSC, config::SolverConfig)

Automatically selects the best solver method based on matrix properties.
"""
function select_solver_method(K::SparseMatrixCSC, config::SolverConfig)
    if config.method != :auto
        return config.method
    end
    
    n = size(K, 1)
    mem_estimates = estimate_memory_usage(K)
    matrix_props = check_matrix_properties(K)
    
    # Konzervativnější rozhodování
    if n < 50000 && mem_estimates[:direct] < config.memory_limit * 0.5
        return :direct
    elseif matrix_props.symmetric && matrix_props.positive_definite
        return :cg
    elseif matrix_props.symmetric
        return :minres
    else
        return mem_estimates[:gmres] < config.memory_limit ? :gmres : :bicgstab
    end
end

"""
    create_preconditioner(K::SparseMatrixCSC, config::SolverConfig)

Creates a preconditioner based on the configuration.
"""
function create_preconditioner(K::SparseMatrixCSC, config::SolverConfig; symmetric::Bool=true)
    if config.preconditioner == :none
        return I
    elseif config.preconditioner == :diagonal
        # Diagonal (Jacobi) preconditioner
        D = diag(K)
        # Ensure no zero diagonal elements
        D[abs.(D) .< 1e-10] .= 1.0
        return Diagonal(1.0 ./ D)
    elseif config.preconditioner == :ilu
        # Incomplete LU factorization
        if symmetric && config.preconditioner == :ichol
            # Try incomplete Cholesky for symmetric matrices
            try
                return CholeskyPreconditioner(K, config.drop_tolerance)
            catch
                config.verbose && println("IChol failed, falling back to ILU")
                return ilu(K, τ = config.drop_tolerance)
            end
        else
            return ilu(K, τ = config.drop_tolerance)
        end
    elseif config.preconditioner == :ichol
        # Incomplete Cholesky for symmetric positive definite
        return CholeskyPreconditioner(K, config.drop_tolerance)
    else
        return I
    end
end

"""
    solve_with_krylov(K, f, method, config, matrix_props)

Solves the system using Krylov.jl methods.
"""
function solve_with_krylov(K, f, method, config, matrix_props)
    n = length(f)
    u = zeros(n)
    
    # Create preconditioner
    P = create_preconditioner(K, config; symmetric=matrix_props.symmetric)
    
    # Set up Krylov solver options
    kwargs = Dict(
        :atol => config.tolerance,
        :rtol => config.tolerance,
        :itmax => config.max_iterations,
        :verbose => config.verbose ? 1 : 0,
        :history => config.history
    )
    
    if method == :cg
        # Conjugate Gradient for SPD matrices
        if P != I
            kwargs[:M] = P  # Preconditioner
        end
        
        solver = CgSolver(n, n, typeof(u))
        u, stats = cg!(solver, K, f; kwargs...)
        
    elseif method == :minres
        # MINRES for symmetric indefinite matrices
        if P != I
            kwargs[:M] = P
        end
        
        solver = MinresSolver(n, n, typeof(u))
        u, stats = minres!(solver, K, f; kwargs...)
        
    elseif method == :gmres
        # GMRES for general matrices
        if P != I
            kwargs[:M] = P
        end
        kwargs[:restart] = config.restart
        
        solver = GmresSolver(n, n, config.restart, typeof(u))
        u, stats = gmres!(solver, K, f; kwargs...)
        
    elseif method == :bicgstab
        # BiCGSTAB for non-symmetric matrices (memory efficient)
        if P != I
            kwargs[:M] = P
        end
        
        solver = BicgstabSolver(n, n, typeof(u))
        u, stats = bicgstab!(solver, K, f; kwargs...)
        
    else
        error("Unknown Krylov method: $method")
    end
    
    # Report results
    if config.verbose
        println("Solver: $(uppercase(string(method)))")
        println("Iterations: $(stats.niter)")
        println("Converged: $(stats.solved)")
        println("Residual: $(stats.residuals[end])")
    end
    
    if !stats.solved
        @warn "Krylov solver did not converge after $(stats.niter) iterations"
    end
    
    return u, stats
end

"""
    solve_with_iterativesolvers(K, f, method, config)

Fallback solver using IterativeSolvers.jl.
"""
function solve_with_iterativesolvers(K, f, method, config)
    n = length(f)
    u = zeros(n)
    
    # Create preconditioner
    P = create_preconditioner(K, config)
    
    if method == :cg
        u, ch = cg!(u, K, f; 
                  Pl = P,
                  tol = config.tolerance,
                  maxiter = config.max_iterations,
                  verbose = config.verbose,
                  log = true)
        
        if config.verbose && ch.isconverged
            println("CG converged in $(ch.iters) iterations")
        end
        
    elseif method == :gmres || method == :minres || method == :bicgstab
        # GMRES for all non-CG cases in IterativeSolvers
        u, ch = gmres!(u, K, f;
                     Pl = P,
                     tol = config.tolerance,
                     restart = config.restart,
                     maxiter = config.max_iterations,
                     verbose = config.verbose,
                     log = true)
        
        if config.verbose && ch.isconverged
            println("GMRES converged in $(ch.iters) iterations")
        end
        
    else
        error("Unknown method for IterativeSolvers: $method")
    end
    
    return u, ch
end

"""
    solve_system_robust(K, f, dh, cellvalues, λ, μ, constraints...; 
                       config::SolverConfig = SolverConfig())

Robust solver for FEM systems with multiple solver options.

Parameters:
- `K`: global stiffness matrix
- `f`: global load vector
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `λ`, `μ`: material parameters
- `constraints...`: ConstraintHandlers with boundary conditions
- `config`: SolverConfig with solver settings

Returns:
- Same as original solve_system function
"""
function solve_system_robust(K, f, dh, cellvalues, λ, μ, constraints...; 
                           config::SolverConfig = SolverConfig())
    
    # Apply constraints
    for ch in constraints
        apply_zero!(K, f, ch)
    end
    
    # Analyze matrix properties
    matrix_props = check_matrix_properties(K)
    
    # Select solver method
    method = select_solver_method(K, config)
    
    # Solve the system
    u = zeros(size(f))
    solve_time = @elapsed begin
        if method == :direct
            config.verbose && println("\nUsing direct solver (backslash)")
            u = K \ f
            
        else
            # Use iterative solver
            if USE_KRYLOV
                config.verbose && println("\nUsing Krylov.jl solver")
                u, stats = solve_with_krylov(K, f, method, config, matrix_props)
            else
                config.verbose && println("\nUsing IterativeSolvers.jl solver")
                u, ch = solve_with_iterativesolvers(K, f, method, config)
            end
        end
    end
    
    config.verbose && println("Solve time: $(round(solve_time, digits=2)) seconds")
    
    # Calculate deformation energy
    deformation_energy = 0.5 * dot(u, K * u)
    
    # Calculate stresses
    stress_field, max_von_mises, max_stress_cell = calculate_stresses(u, dh, cellvalues, λ, μ)
    
    if config.verbose
        println("\nAnalysis complete")
        println("Deformation energy: $deformation_energy J")
        println("Maximum von Mises stress: $max_von_mises at cell $max_stress_cell")
    end
    
    return u, deformation_energy, stress_field, max_von_mises, max_stress_cell
end

"""
    solve_system_robust_simp(K, f, dh, cellvalues, material_model, density_data, constraints...; 
                           config::SolverConfig = SolverConfig())

Robust solver for SIMP-based FEM systems.
"""
function solve_system_robust_simp(K, f, dh, cellvalues, material_model, density_data, constraints...; 
                                config::SolverConfig = SolverConfig())
    
    # Apply constraints
    for ch in constraints
        apply_zero!(K, f, ch)
    end
    
    # Analyze matrix properties
    matrix_props = check_matrix_properties(K)
    
    # Select solver method
    method = select_solver_method(K, config)
    
    # Solve the system
    u = zeros(size(f))
    solve_time = @elapsed begin
        if method == :direct
            config.verbose && println("\nUsing direct solver (backslash)")
            u = K \ f
            
        else
            # Use iterative solver
            if USE_KRYLOV
                config.verbose && println("\nUsing Krylov.jl solver")
                u, stats = solve_with_krylov(K, f, method, config, matrix_props)
            else
                config.verbose && println("\nUsing IterativeSolvers.jl solver")
                u, ch = solve_with_iterativesolvers(K, f, method, config)
            end
        end
    end
    
    config.verbose && println("Solve time: $(round(solve_time, digits=2)) seconds")
    
    # Calculate deformation energy
    deformation_energy = 0.5 * dot(u, K * u)
    
    # Calculate stresses for SIMP
    stress_field, max_von_mises, max_stress_cell = calculate_stresses_simp(u, dh, cellvalues, material_model, density_data)
    
    if config.verbose
        println("\nAnalysis complete")
        println("Deformation energy: $deformation_energy J")
        println("Maximum von Mises stress: $max_von_mises at cell $max_stress_cell")
    end
    
    return u, deformation_energy, stress_field, max_von_mises, max_stress_cell
end

# Export functions
export solve_system_robust, solve_system_robust_simp, SolverConfig

# Convenience presets for common configurations
"""
    SolverConfig_LargeSymmetric()

Preset configuration for large symmetric positive definite problems.
"""
function SolverConfig_LargeSymmetric()
    return SolverConfig(
        method = :cg,
        preconditioner = :ichol,
        tolerance = 1e-8,
        max_iterations = 5000,
        verbose = true
    )
end

"""
    SolverConfig_LargeGeneral()

Preset configuration for large general (possibly non-symmetric) problems.
"""
function SolverConfig_LargeGeneral()
    return SolverConfig(
        method = :gmres,
        preconditioner = :ilu,
        tolerance = 1e-8,
        restart = 50,
        max_iterations = 2000,
        verbose = true
    )
end

"""
    SolverConfig_MemoryEfficient()

Preset configuration for extremely large problems with memory constraints.
"""
function SolverConfig_MemoryEfficient()
    return SolverConfig(
        method = :bicgstab,
        preconditioner = :diagonal,
        tolerance = 1e-7,
        max_iterations = 10000,
        verbose = true
    )
end

export SolverConfig_LargeSymmetric, SolverConfig_LargeGeneral, SolverConfig_MemoryEfficient

# Example usage documentation
"""
    example_usage()

Shows various ways to use the robust solver.

# Examples

## Automatic solver selection
```julia
config = SolverConfig()
u, energy, stress_field, max_vm, max_cell = solve_system_robust(K, f, dh, cellvalues, λ, μ, ch; config=config)
```

## Force CG with IChol preconditioner (for symmetric problems)
```julia
config = SolverConfig(
    method = :cg,
    preconditioner = :ichol,
    tolerance = 1e-10,
    max_iterations = 2000
)
u, energy, stress_field, max_vm, max_cell = solve_system_robust(K, f, dh, cellvalues, λ, μ, ch; config=config)
```

## Use preset for large symmetric problems
```julia
config = SolverConfig_LargeSymmetric()
u, energy, stress_field, max_vm, max_cell = solve_system_robust(K, f, dh, cellvalues, λ, μ, ch; config=config)
```

## Memory-constrained solving
```julia
config = SolverConfig(
    method = :auto,
    memory_limit = 4.0,  # Limit to 4 GB
    preconditioner = :diagonal
)
u, energy, stress_field, max_vm, max_cell = solve_system_robust(K, f, dh, cellvalues, λ, μ, ch; config=config)
```
"""
function example_usage() end
