# SimpleSolverConfig.jl - Jednoduchá konfigurace s dvěma solvery

using IterativeSolvers
using LinearAlgebra
using SparseArrays

"""
Jednoduchý výběr solveru - buď přímý nebo iterativní.
"""
@enum SimpleSolverType begin
    DIRECT     # Cholesky factorization (rychlý, vysoká paměť)
    ITERATIVE  # Conjugate Gradient s předpodmíněním (pomalý, nízká paměť)
end

"""
Jednoduchá konfigurace solveru s minimálními parametry.

# Fields:
- `solver_type`: DIRECT nebo ITERATIVE
- `max_iterations`: Maximum iterací pro iterativní solver (default: 1000)
- `tolerance`: Tolerace konvergence pro iterativní solver (default: 1e-6)
- `verbose`: Tisknout informace o průběhu (default: false)
"""
struct SimpleSolverConfig
    solver_type::SimpleSolverType
    max_iterations::Int
    tolerance::Float64
    verbose::Bool
    
    # Konstruktor s rozumnými defaulty
    function SimpleSolverConfig(solver_type::SimpleSolverType = DIRECT;
                               max_iterations::Int = 1000,
                               tolerance::Float64 = 1e-6,
                               verbose::Bool = false)
        new(solver_type, max_iterations, tolerance, verbose)
    end
end

# Přednastavené konfigurace
"""Rychlý přímý solver - vysoká paměť, rychlé řešení"""
direct_solver() = SimpleSolverConfig(DIRECT, verbose=false)

"""Memory-efficient iterativní solver - nízká paměť, pomalejší"""
iterative_solver(; max_iterations=1000, tolerance=1e-6, verbose=false) = 
    SimpleSolverConfig(ITERATIVE, max_iterations=max_iterations, tolerance=tolerance, verbose=verbose)

"""
    estimate_memory_usage(matrix_size::Int, nnz::Int, solver_type::SimpleSolverType)

Odhad spotřeby paměti pro daný solver.

# Returns:
- Odhad paměti v GB
"""
function estimate_memory_usage(matrix_size::Int, nnz::Int, solver_type::SimpleSolverType)
    # Základní paměť pro uložení sparse matice
    matrix_memory_gb = (nnz * 8 + matrix_size * 8) / 1e9
    
    if solver_type == DIRECT
        # Cholesky faktorizace - přibližně 15-25x původní matice pro 3D FEM
        factorization_memory_gb = matrix_memory_gb * 20
        return matrix_memory_gb + factorization_memory_gb
    else # ITERATIVE
        # CG potřebuje jen několik pracovních vektorů
        vectors_memory_gb = (matrix_size * 8 * 6) / 1e9  # ~6 vektorů velikosti n
        return matrix_memory_gb + vectors_memory_gb
    end
end

"""
    choose_solver_automatically(matrix_size::Int, nnz::Int, available_ram_gb::Float64)

Automaticky vybere vhodný solver podle velikosti problému a dostupné RAM.

# Parameters:
- `matrix_size`: Velikost matice (n pro n×n)
- `nnz`: Počet nenulových prvků
- `available_ram_gb`: Dostupná RAM v GB

# Returns:
- `SimpleSolverConfig` s doporučeným solverem
"""
function choose_solver_automatically(matrix_size::Int, nnz::Int, available_ram_gb::Float64)
    direct_memory = estimate_memory_usage(matrix_size, nnz, DIRECT)
    iterative_memory = estimate_memory_usage(matrix_size, nnz, ITERATIVE)
    
    println("Analýza problému:")
    println("  Velikost matice: $(matrix_size) × $(matrix_size)")
    println("  Nenulové prvky: $(nnz)")
    println("  Dostupná RAM: $(available_ram_gb) GB")
    println("  Odhad paměti - Přímý solver: $(round(direct_memory, digits=2)) GB")
    println("  Odhad paměti - Iterativní solver: $(round(iterative_memory, digits=2)) GB")
    
    # Použij přímý solver pokud se vejde do 70% dostupné RAM
    if direct_memory < available_ram_gb * 0.7
        println("  → Doporučení: Přímý solver (vejde se do paměti)")
        return SimpleSolverConfig(DIRECT, verbose=true)
    else
        println("  → Doporučení: Iterativní solver (úspora paměti)")
        # Pro velké problémy zvyš počet iterací
        max_iter = max(1000, matrix_size ÷ 100)
        return SimpleSolverConfig(ITERATIVE, max_iterations=max_iter, tolerance=1e-8, verbose=true)
    end
end

"""
    solve_linear_system_simple(K, f, config::SimpleSolverConfig)

Řeší lineární systém K*x = f s jednoduchou konfigurací.

# Parameters:
- `K`: Sparse matice tuhosti
- `f`: Vektor pravé strany
- `config`: Konfigurace solveru

# Returns:
- `x`: Řešení
- `info`: Dictionary s informacemi o řešení
"""
function solve_linear_system_simple(K, f, config::SimpleSolverConfig)
    n = size(K, 1)
    
    if config.verbose
        println("Řešení lineárního systému:")
        println("  Velikost: $(n) × $(n)")
        println("  Nenulové prvky: $(nnz(K))")
        println("  Solver: $(config.solver_type)")
    end
    
    if config.solver_type == DIRECT
        return solve_direct_simple(K, f, config)
    else
        return solve_iterative_simple(K, f, config)
    end
end

"""
Přímý solver s Cholesky faktorizací (pro SPD matice) nebo LU jako fallback.
"""
function solve_direct_simple(K, f, config::SimpleSolverConfig)
    if config.verbose
        println("  Používám přímý solver...")
    end
    
    start_time = time()
    
    # Zkus Cholesky faktorizaci (efektivnější pro SPD matice)
    try
        if config.verbose
            println("  Pokus o Cholesky faktorizaci...")
        end
        F = cholesky(K)
        x = F \ f
        solve_time = time() - start_time
        
        info = Dict(
            "converged" => true,
            "method" => "Cholesky",
            "solve_time" => solve_time,
            "residual_norm" => norm(K * x - f)
        )
        
        if config.verbose
            println("  ✓ Cholesky faktorizace úspěšná")
            println("  ✓ Čas řešení: $(round(solve_time, digits=2)) s")
        end
        
        return x, info
        
    catch e
        if config.verbose
            println("  Cholesky selhala, zkouším LU faktorizaci...")
        end
        
        # Fallback na LU faktorizaci
        try
            x = K \ f
            solve_time = time() - start_time
            
            info = Dict(
                "converged" => true,
                "method" => "LU",
                "solve_time" => solve_time,
                "residual_norm" => norm(K * x - f)
            )
            
            if config.verbose
                println("  ✓ LU faktorizace úspěšná")
                println("  ✓ Čas řešení: $(round(solve_time, digits=2)) s")
            end
            
            return x, info
            
        catch e2
            error("Oba přímé solvery selhaly: Cholesky: $e, LU: $e2")
        end
    end
end

"""
Iterativní solver s Conjugate Gradient a diagonálním předpodmíněním.
FIXED: Corrected IterativeSolvers API usage
"""
function solve_iterative_simple(K, f, config::SimpleSolverConfig)
    if config.verbose
        println("  Používám iterativní solver (Conjugate Gradient)...")
    end
    
    start_time = time()
    
    # Jednoduché diagonální předpodmínění
    # Vezmi reciproké hodnoty diagonály jako předpodmínění
    diag_K = diag(K)
    # Nahraď nuly malými hodnotami aby se předešlo dělení nulou
    diag_K[diag_K .== 0] .= 1e-12
    preconditioner = Diagonal(1 ./ sqrt.(abs.(diag_K)))
    
    # Počáteční odhad (nulový vektor)
    x0 = zeros(size(f))
    
    # FIXED: Correct IterativeSolvers API usage
    # Use cg! (in-place) with proper parameter names
    try
        # Copy initial guess to solution vector
        x = copy(x0)
        
        # Solve using CG with preconditioner - use cg! for in-place modification
        history = cg!(x, K, f;
                     Pl = preconditioner,                # Left preconditioner
                     maxiter = config.max_iterations,    # Maximum iterations
                     reltol = config.tolerance,          # Relative tolerance (FIXED: use reltol instead of tol)
                     verbose = config.verbose,           # Print progress
                     log = true                          # Return convergence log
                    )
        
        solve_time = time() - start_time
        
        info = Dict(
            "converged" => history.converged,
            "method" => "Conjugate Gradient",
            "iterations" => history.iters,
            "solve_time" => solve_time,
            "residual_norm" => history.residuals[end],
            "residual_history" => history.residuals
        )
        
        if config.verbose
            if history.converged
                println("  ✓ Konvergence za $(history.iters) iterací")
            else
                println("  ✗ Nekonvergoval za $(config.max_iterations) iterací")
            end
            println("  ✓ Finální reziduum: $(history.residuals[end])")
            println("  ✓ Čas řešení: $(round(solve_time, digits=2)) s")
        end
        
        return x, info
        
    catch e
        if config.verbose
            println("  ✗ CG solver selhal: $e")
            println("  Zkouším fallback na přímý solver...")
        end
        
        # Fallback to direct solver if iterative fails
        return solve_direct_simple(K, f, config)
    end
end

"""
    auto_solve(K, f; available_ram_gb::Float64 = 0.0, verbose::Bool = false)

Nejjednodušší interface - automaticky vybere a použije vhodný solver.

# Parameters:
- `K`: Matice tuhosti  
- `f`: Vektor pravé strany
- `available_ram_gb`: Dostupná RAM v GB (0 = auto-detekce)
- `verbose`: Tisknout informace

# Returns:
- `x`: Řešení
- `info`: Informace o řešení
"""
function auto_solve(K, f; available_ram_gb::Float64 = 0.0, verbose::Bool = false)
    matrix_size = size(K, 1)
    nnz_count = nnz(K)
    
    # Auto-detekce dostupné RAM
    if available_ram_gb <= 0.0
        total_ram_gb = Sys.total_memory() / 1e9
        available_ram_gb = total_ram_gb * 0.7  # Použij 70% celkové RAM
    end
    
    # Automatický výběr solveru
    config = choose_solver_automatically(matrix_size, nnz_count, available_ram_gb)
    
    if verbose
        config = SimpleSolverConfig(config.solver_type, 
                                  max_iterations=config.max_iterations,
                                  tolerance=config.tolerance, 
                                  verbose=true)
    end
    
    # Řešení
    return solve_linear_system_simple(K, f, config)
end
