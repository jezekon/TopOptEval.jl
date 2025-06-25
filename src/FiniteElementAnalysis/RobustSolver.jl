"""
Enhanced solver module for large-scale FEM problems with Krylov methods
and memory-efficient strategies.

This module provides robust solving capabilities for FEM systems using
modern Krylov subspace methods with various preconditioning options.
All iterative methods are implemented using Krylov.jl for optimal performance.
"""

using LinearAlgebra
using SparseArrays
using Preconditioners
using Printf

# Load Krylov.jl for iterative solving methods
try
    using Krylov
    import Krylov: gmres, cg, minres, bicgstab
catch
    error("Krylov.jl must be installed for iterative solver functionality")
end

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

# Default configuration constructor
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
Returns a Dict with memory estimates in GB for direct and iterative methods.
"""
function estimate_memory_usage(K::SparseMatrixCSC)
    n = size(K, 1)
    nnz_K = nnz(K)
    
    # Basic memory requirements for matrix storage
    matrix_memory = (nnz_K * 8 + n * 8) / 1e9  # 8 bytes per float64
    vector_memory = n * 8 / 1e9
    
    # Conservative fill-in factor estimates for large FEM matrices
    # Use realistic estimates instead of optimistic 10% fill-in
    if n > 500000
        fill_factor = min(50.0, n / 20000)  # 50x-100x for very large matrices
    elseif n > 100000
        fill_factor = min(20.0, n / 10000)  # 20x-50x for medium matrices
    else
        fill_factor = 5.0  # 5x for small matrices
    end
    
    # Memory estimates for different solver types
    direct_memory = matrix_memory * (1 + fill_factor)
    cg_memory = matrix_memory + 6 * vector_memory      # CG needs ~6 vectors
    gmres_memory = matrix_memory + 35 * vector_memory  # GMRES needs more storage
    
    return Dict(
        :direct => direct_memory,
        :cg => cg_memory,
        :gmres => gmres_memory,
        :matrix_only => matrix_memory
    )
end

"""
    estimate_bandwidth(K::SparseMatrixCSC)

Estimates the bandwidth of a sparse matrix by checking maximum distance
between row and column indices for nonzero elements.
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
Checks symmetry and positive definiteness (heuristically).
"""
function check_matrix_properties(K::SparseMatrixCSC)
    n = size(K, 1)
    
    # Check symmetry efficiently for sparse matrices (sample-based)
    is_symmetric = true
    rows = rowvals(K)
    vals = nonzeros(K)
    
    # Sample check for symmetry (full check would be too expensive)
    sample_size = min(100, n ÷ 10)
    for j in 1:sample_size:n
        for idx in nzrange(K, j)
            i = rows[idx]
            if i != j  # Skip diagonal elements
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
    
    # Check positive definiteness heuristically by examining diagonal
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

Automatically selects the best solver method based on matrix size,
memory requirements, and mathematical properties.
"""
function select_solver_method(K::SparseMatrixCSC, config::SolverConfig)
    if config.method != :auto
        return config.method
    end
    
    n = size(K, 1)
    mem_estimates = estimate_memory_usage(K)
    matrix_props = check_matrix_properties(K)
    
    # Conservative decision making for solver selection
    if n < 50000 && mem_estimates[:direct] < config.memory_limit * 0.5
        return :direct
    elseif matrix_props.symmetric && matrix_props.positive_definite
        return :cg  # Conjugate gradient for SPD matrices
    elseif matrix_props.symmetric
        return :minres  # MINRES for symmetric indefinite
    else
        # Choose between GMRES and BiCGSTAB based on memory
        return mem_estimates[:gmres] < config.memory_limit ? :gmres : :bicgstab
    end
end

"""
    create_preconditioner(K::SparseMatrixCSC, config::SolverConfig; symmetric::Bool=true)

Creates a preconditioner based on configuration settings with robust fallback options.
"""
function create_preconditioner(K::SparseMatrixCSC, config::SolverConfig; symmetric::Bool=true)
    if config.preconditioner == :none
        return I
        
    elseif config.preconditioner == :diagonal
        # Diagonal (Jacobi) preconditioner - always safe and efficient
        D = diag(K)
        # Ensure no zero or near-zero diagonal elements
        D[abs.(D) .< 1e-12] .= 1.0
        return Diagonal(1.0 ./ D)
        
    elseif config.preconditioner == :ilu
        # Incomplete LU factorization with fallback to diagonal
        try
            return ilu(K, τ = config.drop_tolerance)
        catch e
            config.verbose && @warn "ILU preconditioner failed, falling back to diagonal: $e"
            D = diag(K)
            D[abs.(D) .< 1e-12] .= 1.0
            return Diagonal(1.0 ./ D)
        end
        
    elseif config.preconditioner == :ichol
        # Incomplete Cholesky for symmetric positive definite matrices
        try
            if symmetric
                return CholeskyPreconditioner(K, config.drop_tolerance)
            else
                # Fall back to ILU for non-symmetric matrices
                return ilu(K, τ = config.drop_tolerance)
            end
        catch e
            config.verbose && @warn "IChol preconditioner failed, falling back to diagonal: $e"
            D = diag(K)
            D[abs.(D) .< 1e-12] .= 1.0
            return Diagonal(1.0 ./ D)
        end
        
    else
        @warn "Unknown preconditioner type: $(config.preconditioner), using identity"
        return I
    end
end

"""
    solve_with_krylov(K, f, method, config, matrix_props)

Solves the linear system using Krylov.jl methods with comprehensive
verbose output and robust error handling.
"""
function solve_with_krylov(K, f, method, config, matrix_props)
    n = length(f)
    u = zeros(n)
    
    # Create preconditioner with proper error handling
    P = try
        create_preconditioner(K, config; symmetric=matrix_props.symmetric)
    catch e
        if config.verbose
            @warn "Preconditioner creation failed, falling back to no preconditioning: $e"
        end
        P = I
    end
    
    # Set up Krylov solver options - enhanced for better monitoring
    kwargs = Dict{Symbol, Any}(
        :atol => config.tolerance,
        :rtol => config.tolerance,  
        :itmax => config.max_iterations,
        :verbose => config.verbose ? 2 : 0,
        :history => true
    )
    
    # Add preconditioner if available
    if P != I
        kwargs[:M] = P
    end
    
    # Print detailed initial information
    if config.verbose
        println("\n" * "="^60)
        println("KRYLOV SOLVER DETAILS")
        println("="^60)
        println("Method: $(uppercase(string(method)))")
        println("Matrix size: $(n) × $(n)")
        println("Non-zeros: $(nnz(K))")
        println("Preconditioner: $(config.preconditioner)")
        println("Target tolerance: $(config.tolerance)")
        println("Max iterations: $(config.max_iterations)")
        if method == :gmres
            println("GMRES restart: $(config.restart)")
        end
        println("Symmetric: $(matrix_props.symmetric)")
        println("Positive definite: $(matrix_props.positive_definite)")
        println("-"^60)
        println("Starting iterations...")
    end
    
    # Solve based on method with enhanced error handling
    local stats
    local success = false
    
    try
        if method == :cg
            # Conjugate Gradient for symmetric positive definite matrices
            if config.verbose
                println("Using CG (Conjugate Gradient) solver...")
            end
            u, stats = cg(K, f; kwargs...)
            success = true
            
        elseif method == :minres
            # MINRES for symmetric indefinite matrices  
            if config.verbose
                println("Using MINRES solver...")
            end
            u, stats = minres(K, f; kwargs...)
            success = true
            
        elseif method == :gmres
            # GMRES for general matrices
            if config.verbose
                println("Using GMRES solver...")
            end
            # Handle restart parameter carefully
            gmres_kwargs = copy(kwargs)
            try
                # Try with restart parameter first
                gmres_kwargs[:restart] = config.restart
                u, stats = gmres(K, f; gmres_kwargs...)
                success = true
            catch e
                if config.verbose
                    @warn "GMRES with restart parameter failed, trying without: $e"
                end
                # Remove restart parameter and try again
                delete!(gmres_kwargs, :restart)
                u, stats = gmres(K, f; gmres_kwargs...)
                success = true
            end
            
        elseif method == :bicgstab
            # BiCGSTAB for non-symmetric matrices (memory efficient)
            if config.verbose
                println("Using BiCGSTAB solver...")
            end
            u, stats = bicgstab(K, f; kwargs...)
            success = true
            
        else
            error("Unknown Krylov method: $method")
        end
        
    catch e
        success = false
        if config.verbose
            @warn "Primary Krylov method $method failed: $e"
            println("Attempting fallback to simple CG...")
        end
        
        # Fallback to simple CG without preconditioner
        simple_kwargs = Dict{Symbol, Any}(
            :atol => config.tolerance * 10,  # Relax tolerance for fallback
            :rtol => config.tolerance * 10,
            :itmax => config.max_iterations,
            :verbose => config.verbose ? 2 : 0,
            :history => true
        )
        
        try
            u, stats = cg(K, f; simple_kwargs...)
            success = true
            if config.verbose
                println("Fallback CG solver succeeded!")
            end
        catch e2
            @error "Both primary and fallback solvers failed: $e2"
            # Last resort: direct solver if matrix is small enough
            if n < 100000
                @warn "Attempting direct solve as last resort..."
                u = K \ f
                success = true
                # Create dummy stats for consistency
                stats = (solved = true, niter = 1, residual = [0.0])
            else
                rethrow(e2)
            end
        end
    end
    
    # Enhanced reporting with convergence analysis
    if config.verbose
        println("-"^60)
        println("SOLVER RESULTS")
        println("-"^60)
        
        if hasfield(typeof(stats), :niter)
            println("Iterations completed: $(stats.niter)")
            
            # Print convergence progress every N iterations for monitoring
            if hasfield(typeof(stats), :residual) && length(stats.residual) > 1
                println("\nConvergence history (every 50th iteration):")
                for i in 1:50:length(stats.residual)
                    @printf("  Iteration %5d: residual = %.6e\n", i, stats.residual[i])
                end
                # Always show the last iteration
                last_iter = length(stats.residual)
                if last_iter % 50 != 1
                    @printf("  Iteration %5d: residual = %.6e\n", last_iter, stats.residual[end])
                end
            end
        end
        
        if hasfield(typeof(stats), :solved)
            converged_status = stats.solved ? "✓ CONVERGED" : "✗ NOT CONVERGED"
            println("\nStatus: $converged_status")
        end
        
        # Try to get final residual from different possible fields
        final_residual = nothing
        if hasfield(typeof(stats), :residual) && length(stats.residual) > 0
            final_residual = stats.residual[end]
        elseif hasfield(typeof(stats), :residuals) && length(stats.residuals) > 0
            final_residual = stats.residuals[end]
        elseif hasfield(typeof(stats), :resnorm)
            final_residual = stats.resnorm
        end
        
        if final_residual !== nothing
            println("Final residual: $(final_residual)")
            println("Target tolerance: $(config.tolerance)")
            println("Convergence ratio: $(final_residual / config.tolerance)")
        end
        
        # Calculate actual residual for verification
        actual_residual = norm(K * u - f)
        println("Actual residual ||Ku - f||: $(actual_residual)")
        
        # Convergence quality assessment
        if actual_residual < config.tolerance
            println("✓ Solution satisfies tolerance requirement")
        elseif actual_residual < config.tolerance * 100
            println("⚠ Solution is reasonably accurate")
        else
            println("✗ Solution may be inaccurate")
        end
        
        println("="^60)
    end
    
    # Enhanced warning for non-convergence with suggestions
    if hasfield(typeof(stats), :solved) && !stats.solved
        actual_residual = norm(K * u - f)
        final_residual_val = final_residual !== nothing ? final_residual : actual_residual
        niter_val = hasfield(typeof(stats), :niter) ? stats.niter : "unknown"
        
        @warn """
        Krylov solver did not converge after $niter_val iterations.
        
        Final residual: $final_residual_val
        Actual residual: $actual_residual
        Target tolerance: $(config.tolerance)
        
        Suggestions:
        1. Increase max_iterations (current: $(config.max_iterations))
        2. Relax tolerance (try $(config.tolerance * 100))
        3. Try different preconditioner (current: $(config.preconditioner))
        4. Try different method (current: $method, try :cg or :bicgstab)
        5. Check matrix conditioning
        6. Verify boundary conditions are properly applied
        """
    end
    
    return u, stats
end

"""
    solve_system_robust(K, f, dh, cellvalues, λ, μ, constraints...; 
                       config::SolverConfig = SolverConfig())

Robust solver for FEM systems with automatic method selection and comprehensive
error handling. Uses only Krylov.jl for iterative methods.

Parameters:
- `K`: global stiffness matrix
- `f`: global load vector
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `λ`, `μ`: material parameters
- `constraints...`: ConstraintHandlers with boundary conditions
- `config`: SolverConfig with solver settings

Returns:
- Same as original solve_system function: (u, energy, stress_field, max_von_mises, max_stress_cell)
"""
function solve_system_robust(K, f, dh, cellvalues, λ, μ, constraints...; 
                           config::SolverConfig = SolverConfig())
    
    # Apply all constraint handlers to the system
    for ch in constraints
        apply_zero!(K, f, ch)
    end
    
    # Analyze matrix properties for optimal solver selection
    matrix_props = check_matrix_properties(K)
    
    # Select appropriate solver method
    method = select_solver_method(K, config)
    
    # Enhanced pre-solve diagnostics
    if config.verbose
        println("\n" * "="^60)
        println("PRE-SOLVE DIAGNOSTICS")
        println("="^60)
        
        # Memory estimates
        mem_est = estimate_memory_usage(K)
        println("Memory estimates:")
        println("  Matrix storage: $(round(mem_est[:matrix_only], digits=2)) GB")
        println("  Direct solver: $(round(mem_est[:direct], digits=2)) GB")
        println("  CG solver: $(round(mem_est[:cg], digits=2)) GB")
        println("  GMRES solver: $(round(mem_est[:gmres], digits=2)) GB")
        
        # Matrix condition estimation (for small matrices)
        if size(K, 1) < 10000
            try
                κ = cond(Matrix(K))
                println("Condition number: $(round(κ, digits=2))")
                if κ > 1e12
                    println("⚠ Matrix is ill-conditioned - convergence may be slow")
                end
            catch
                println("Condition number: Could not compute")
            end
        end
        
        println("Selected method: $(uppercase(string(method)))")
    end
    
    # Solve the system using appropriate method
    u = zeros(size(f))
    solve_time = @elapsed begin
        if method == :direct
            config.verbose && println("\nUsing direct solver (backslash)")
            u = K \ f
            
        else
            # Use Krylov iterative solver with enhanced monitoring
            config.verbose && println("\nUsing Krylov.jl iterative solver")
            u, stats = solve_with_krylov(K, f, method, config, matrix_props)
        end
    end
    
    if config.verbose
        println("\nSolve time: $(round(solve_time, digits=2)) seconds")
        println("Solution vector norm: $(norm(u))")
    end
    
    config.verbose && println("Solve time: $(round(solve_time, digits=2)) seconds")
    
    # Calculate deformation energy: U = 0.5 * u^T * K * u
    deformation_energy = 0.5 * dot(u, K * u)
    
    # Calculate stress field from displacement solution
    stress_field, max_von_mises, max_stress_cell = calculate_stresses(u, dh, cellvalues, λ, μ)
    
    if config.verbose
        println("\n" * "="^60)
        println("FINAL ANALYSIS RESULTS")
        println("="^60)
        println("Deformation energy: $(round(deformation_energy, digits=6)) J")
        println("Maximum von Mises stress: $(round(max_von_mises, digits=2)) Pa")
        println("Max stress location: cell $(max_stress_cell)")
        println("="^60)
    end
    
    return u, deformation_energy, stress_field, max_von_mises, max_stress_cell
end

"""
    solve_system_robust_simp(K, f, dh, cellvalues, material_model, density_data, constraints...; 
                           config::SolverConfig = SolverConfig())

Robust solver for SIMP-based FEM systems with variable material properties.
Uses only Krylov.jl for iterative methods.

Parameters:
- `K`: global stiffness matrix
- `f`: global load vector  
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `material_model`: Function mapping density to material parameters (λ, μ)
- `density_data`: Vector with density values for each cell
- `constraints...`: ConstraintHandlers with boundary conditions
- `config`: SolverConfig with solver settings

Returns:
- Same as solve_system_simp: (u, energy, stress_field, max_von_mises, max_stress_cell)
"""
function solve_system_robust_simp(K, f, dh, cellvalues, material_model, density_data, constraints...; 
                                config::SolverConfig = SolverConfig())
    
    # Apply all constraint handlers to the system
    for ch in constraints
        apply_zero!(K, f, ch)
    end
    
    # Analyze matrix properties for optimal solver selection
    matrix_props = check_matrix_properties(K)
    
    # Select appropriate solver method
    method = select_solver_method(K, config)
    
    # Enhanced pre-solve diagnostics
    if config.verbose
        println("\n" * "="^60)
        println("PRE-SOLVE DIAGNOSTICS (SIMP)")
        println("="^60)
        
        # Memory estimates
        mem_est = estimate_memory_usage(K)
        println("Memory estimates:")
        println("  Matrix storage: $(round(mem_est[:matrix_only], digits=2)) GB")
        println("  Direct solver: $(round(mem_est[:direct], digits=2)) GB")
        println("  CG solver: $(round(mem_est[:cg], digits=2)) GB")
        println("  GMRES solver: $(round(mem_est[:gmres], digits=2)) GB")
        
        # Matrix condition estimation (for small matrices)
        if size(K, 1) < 10000
            try
                κ = cond(Matrix(K))
                println("Condition number: $(round(κ, digits=2))")
                if κ > 1e12
                    println("⚠ Matrix is ill-conditioned - convergence may be slow")
                end
            catch
                println("Condition number: Could not compute")
            end
        end
        
        println("Selected method: $(uppercase(string(method)))")
    end
    
    # Solve the system using appropriate method
    u = zeros(size(f))
    solve_time = @elapsed begin
        if method == :direct
            config.verbose && println("\nUsing direct solver (backslash)")
            u = K \ f
            
        else
            # Use Krylov iterative solver with enhanced monitoring
            config.verbose && println("\nUsing Krylov.jl iterative solver")
            u, stats = solve_with_krylov(K, f, method, config, matrix_props)
        end
    end
    
    if config.verbose
        println("\nSolve time: $(round(solve_time, digits=2)) seconds")
        println("Solution vector norm: $(norm(u))")
    end
    
    # Calculate deformation energy: U = 0.5 * u^T * K * u
    deformation_energy = 0.5 * dot(u, K * u)
    
    # Calculate stress field for SIMP with variable material properties
    stress_field, max_von_mises, max_stress_cell = calculate_stresses_simp(u, dh, cellvalues, material_model, density_data)
    
    if config.verbose
        println("\n" * "="^60)
        println("FINAL ANALYSIS RESULTS (SIMP)")
        println("="^60)
        println("Deformation energy: $(round(deformation_energy, digits=6)) J")
        println("Maximum von Mises stress: $(round(max_von_mises, digits=2)) Pa")
        println("Max stress location: cell $(max_stress_cell)")
        println("="^60)
    end
    
    return u, deformation_energy, stress_field, max_von_mises, max_stress_cell
end

# Export main functions
export solve_system_robust, solve_system_robust_simp, SolverConfig

# Convenience preset configurations for common use cases

"""
    SolverConfig_LargeSymmetric()

Preset configuration optimized for large symmetric positive definite problems.
Uses CG with incomplete Cholesky preconditioning.
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

Preset configuration optimized for large general (possibly non-symmetric) problems.
Uses GMRES with ILU preconditioning and increased restart parameter.
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

Preset configuration for extremely large problems with strict memory constraints.
Uses BiCGSTAB with diagonal preconditioning for minimal memory footprint.
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

# Export convenience configurations
export SolverConfig_LargeSymmetric, SolverConfig_LargeGeneral, SolverConfig_MemoryEfficient
