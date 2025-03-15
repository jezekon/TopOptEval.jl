# Load necessary packages
using Ferrite
using LinearAlgebra
using SparseArrays
using WriteVTK
using StaticArrays
using FerriteGmsh  # Required for parsing GMSH (.msh) files

"""
    create_material_model(youngs_modulus::Float64, poissons_ratio::Float64)

Vytvoří materiálové konstanty pro lineárně elastický materiál.

Parametry:
- `youngs_modulus`: Youngův modul v Pa
- `poissons_ratio`: Poissonovo číslo

Vrací:
- lambda a mí koeficienty pro Hookeův zákon
"""
function create_material_model(youngs_modulus::Float64, poissons_ratio::Float64)
    # Lamého koeficienty
    λ = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    μ = youngs_modulus / (2 * (1 + poissons_ratio))
    
    return λ, μ
end

"""
    constitutive_relation(ε, λ, μ)

Aplikuje lineárně elastický vztah mezi deformací a napětím (Hookeův zákon).

Parametry:
- `ε`: tenzor deformace
- `λ`: první Lamého koeficient
- `μ`: druhý Lamého koeficient (smykový modul)

Vrací:
- tenzor napětí
"""
function constitutive_relation(ε, λ, μ)
    # Lineární elasticita: σ = λ*tr(ε)*I + 2μ*ε
    return λ * tr(ε) * one(ε) + 2μ * ε
end

"""
    assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)

Sestaví globální matici tuhosti a inicializuje vektor zatížení.

Parametry:
- `K`: globální matice tuhosti (modifikovaná in-place)
- `f`: globální vektor zatížení (modifikovaný in-place)
- `dh`: DofHandler
- `cellvalues`: CellValues pro interpolaci a integraci
- `λ`, `μ`: materiálové parametry

Vrací:
- nic (modifikuje K a f in-place)
"""
function assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)
    # Element stiffness matrix and internal force vector
    dim = 3  # 3D problem
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
end

"""
    apply_boundary_conditions!(K, f, grid, dh)

Aplikuje okrajové podmínky - fixace a zatížení.

Parametry:
- `K`: globální matice tuhosti
- `f`: globální vektor zatížení
- `grid`: výpočetní síť
- `dh`: DofHandler

Vrací:
- ConstraintHandler pro další použití
"""
function apply_boundary_conditions!(K, f, grid, dh)
    # Extract problem dimensions
    num_nodes = getnnodes(grid)
    dim = 3  # 3D problem
    
    # 1. Fixed nodes at x=0 (with tolerance 10^-6)
    tol = 1e-6  # Tolerance for floating point comparison
    fixed_nodes = Set{Int}()
    
    # Find nodes on yz plane (x ≈ 0)
    for node_id in 1:num_nodes
        coord = grid.nodes[node_id].x
        if abs(coord[1]) < tol
            push!(fixed_nodes, node_id)
        end
    end
    
    println("Number of fixed nodes: $(length(fixed_nodes))")
    
    # Create and apply Dirichlet boundary conditions for fixed nodes
    ch = ConstraintHandler(dh)
    # Oprava: místo vektoru komponent zde použijeme skalární zápis pro každou komponentu
    for d in 1:dim
        dbc = Dirichlet(:u, fixed_nodes, (x, t) -> 0.0, d)
        add!(ch, dbc)
    end
    close!(ch)
    update!(ch, 0.0)
    apply!(K, f, ch)
    
    # 2. Force of 1N applied at x=60 (with tolerance 10^-6)
    load_nodes = Int[]
    
    # Find nodes on yz plane where x ≈ 60
    for node_id in 1:num_nodes
        coord = grid.nodes[node_id].x
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
    force_per_node = 1.0 / length(load_nodes)
    
    # V nové verzi Ferrite.jl přistupujeme k DOF uzlů jinak
    # Nejprve vytvoříme set DOF pro každý uzel
    node_to_dofs = Dict{Int, Vector{Int}}()
    
    # Pro každou buňku získáme mapování uzlů na DOF
    for cell in CellIterator(dh)
        cell_nodes = getnodes(cell)
        cell_dofs = celldofs(cell)
        
        # Předpokládáme, že pro každý uzel máme 'dim' DOF (pro každý směr jeden)
        # a že jsou uspořádány postupně pro každý uzel
        nodes_per_cell = length(cell_nodes)
        dofs_per_node = length(cell_dofs) ÷ nodes_per_cell
        
        # Pro každý uzel v buňce
        for (local_node_idx, global_node_idx) in enumerate(cell_nodes)
            # Vypočítáme rozsah DOF pro tento uzel v rámci buňky
            start_dof = (local_node_idx - 1) * dofs_per_node + 1
            end_dof = local_node_idx * dofs_per_node
            local_dofs = cell_dofs[start_dof:end_dof]
            
            # Přidáme DOF do slovníku
            if !haskey(node_to_dofs, global_node_idx)
                node_to_dofs[global_node_idx] = local_dofs
            end
        end
    end
    
    # Nyní aplikujeme sílu na uzly, které jsou na zatížené hraně
    for node_id in load_nodes
        if haskey(node_to_dofs, node_id)
            # První DOF pro uzel je ve směru x
            x_dof = node_to_dofs[node_id][1]
            f[x_dof] += force_per_node
        end
    end
    
    return ch
end

"""
    solve_system(K, f, ch)

Řeší systém lineárních rovnic.

Parametry:
- `K`: globální matice tuhosti
- `f`: globální vektor zatížení
- `ch`: ConstraintHandler s okrajovými podmínkami

Vrací:
- vektor posunutí
"""
function solve_system(K, f, ch)
    # Apply zero value to constrained dofs
    apply_zero!(K, f, ch)
    
    # Solve
    u = K \ f  # Implicit method using backslash operator
    
    # Calculate deformation energy: U = 0.5 * u^T * K * u
    deformation_energy = 0.5 * dot(u, K * u)
    
    println("Analysis complete.")
    println("Deformation energy: $deformation_energy J")
    
    return u, deformation_energy
end

"""
    analyze_cantilever_beam(
        grid::Grid,
        youngs_modulus::Float64=210.0e9, 
        poissons_ratio::Float64=0.3
    )

Provede analýzu konzolového nosníku diskretizovaného tetrahedrálními prvky.

Parametry:
- `grid`: Ferrite Grid obsahující síť
- `youngs_modulus`: Youngův modul materiálu v Pa (default: 210 GPa, ocel)
- `poissons_ratio`: Poissonovo číslo materiálu (default: 0.3, ocel)

Vrací:
- Trojici (vektor posunutí, deformační energie, dof_handler)
"""
function analyze_cantilever_beam(
    grid::Grid,
    youngs_modulus::Float64=210.0e9, 
    poissons_ratio::Float64=0.3
)
    # Extract problem dimensions
    num_nodes = getnnodes(grid)
    num_cells = getncells(grid)
    dim = 3  # 3D problem
    
    println("Problem size: $num_nodes nodes, $num_cells tetrahedral elements")
    
    # Create material model
    λ, μ = create_material_model(youngs_modulus, poissons_ratio)
    
    # Create the finite element space
    # Použití správné konstrukce pro Lagrangeovu interpolaci
    ip = Lagrange{RefTetrahedron, 1}()^dim  # vektorová interpolace
    
    # Create quadrature rule
    qr = QuadratureRule{RefTetrahedron}(2)
    
    # Create cell values
    cellvalues = CellValues(qr, ip)
    
    # Set up the FE problem
    dh = DofHandler(grid)
    add!(dh, :u, ip)  # přidání pole posunutí
    close!(dh)
    
    # Allocate solution vectors and system matrices
    n_dofs = ndofs(dh)
    println("Number of DOFs: $n_dofs")
    K = allocate_matrix(dh)
    f = zeros(n_dofs)
    
    # Assemble stiffness matrix
    assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)
    
    # Apply boundary conditions
    ch = apply_boundary_conditions!(K, f, grid, dh)
    
    # Solve the system
    println("Solving linear system with $(n_dofs) degrees of freedom...")
    u, deformation_energy = solve_system(K, f, ch)
    
    return u, deformation_energy, dh
end

"""
    import_and_analyze_mesh(
        mesh_file::String, 
        youngs_modulus::Float64=210.0e9, 
        poissons_ratio::Float64=0.3
    )

Importuje soubor sítě a provede analýzu konzolového nosníku.
Momentálně podporuje pouze soubory .msh z GMSH.

Parametry:
- `mesh_file`: Cesta k souboru GMSH .msh
- `youngs_modulus`: Youngův modul materiálu v Pa (default: 210 GPa, ocel)
- `poissons_ratio`: Poissonovo číslo materiálu (default: 0.3, ocel)

Vrací:
- Trojici (vektor posunutí, deformační energie, dof_handler)
"""
function import_and_analyze_mesh(mesh_file::String, youngs_modulus::Float64=210.0e9, poissons_ratio::Float64=0.3)
    # Check file extension
    ext = lowercase(splitext(mesh_file)[2])
    
    if ext != ".msh"
        error("Unsupported mesh format: $ext. Only .msh format is supported.")
    end
    
    println("Importing mesh from $mesh_file...")
    
    # Import the GMSH .msh file using FerriteGmsh
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

Exportuje výsledky FEM analýzy do VTK souboru pro vizualizaci.

Parametry:
- `displacements`: Vektor řešení obsahující posunutí
- `dh`: DofHandler použitý při analýze
- `output_file`: Cesta k výstupnímu souboru
"""
function export_results(displacements::Vector{Float64}, dh::DofHandler, output_file::String)
    println("Exporting results to $output_file...")
    
    # Create a VTK file to visualize the results
    vtk_file = VTKGridFile(output_file, dh)
    
    # Add displacement field to the VTK file
    write_solution(vtk_file, dh, displacements)
    
    # Write the results
    close(vtk_file)
    println("Results exported successfully.")
end

"""
    main()

Hlavní funkce pro analýzu konzolového nosníku s tetrahedrálními prvky.
"""
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


    using Ferrite
    using FerriteGmsh  # Required for parsing GMSH (.msh) files
    using ReadVTK      # Required for parsing VTK XML formats (.vtu files)
    export import_mesh
    
    """
        import_mesh(mesh_file::String)
    
    Imports a mesh file and returns the Ferrite Grid object.
    Currently supports GMSH (.msh) and VTK XML UnstructuredGrid (.vtu) files.
    
    Parameters:
    - `mesh_file`: Path to the mesh file (.msh or .vtu)
    
    Returns:
    - A Ferrite Grid object
    """
    function import_mesh(mesh_file::String)
        # Check file extension
        ext = lowercase(splitext(mesh_file)[2])
        
        if ext == ".msh"
            println("Importing GMSH mesh from $mesh_file...")
            
            # Import the GMSH .msh file using FerriteGmsh
            grid = FerriteGmsh.togrid(mesh_file)
            
        elseif ext == ".vtu"
            println("Importing VTU mesh from $mesh_file...")
            
            # Read the VTU file using ReadVTK
            vtk_file = ReadVTK.VTKFile(mesh_file)
            
            # Extract coordinates from points
            points = ReadVTK.get_points(vtk_file)
            
            # Extract cells
            vtk_cells = ReadVTK.get_cells(vtk_file)
            
            # Create Ferrite nodes from points (transpose to iterate over columns)
            nodes = [Ferrite.Node(Vec{3}(points[:, i])) for i in 1:size(points, 2)]
            
            # Create Ferrite cells from VTK cells
            ferrite_cells = Ferrite.AbstractCell[]
            
            # Process connectivity, offsets, and types from vtk_cells
            connectivity = vtk_cells.connectivity
            offsets = vtk_cells.offsets
            types = vtk_cells.types
            
            # Create start offsets for cell connectivity indexing
            start_indices = vcat(1, offsets[1:end-1] .+ 1)
            
            for i in 1:length(types)
                # Get connectivity indices for this cell
                conn_indices = start_indices[i]:offsets[i]
                cell_conn = connectivity[conn_indices]
                
                # Map VTK cell types to Ferrite cell types
                # Reference: https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
                vtk_type = types[i]
                
                # VTK_TETRA = 10
                if vtk_type == 10
                    push!(ferrite_cells, Ferrite.Tetrahedron(Tuple(cell_conn)))
                # VTK_HEXAHEDRON = 12
                elseif vtk_type == 12
                    push!(ferrite_cells, Ferrite.Hexahedron(Tuple(cell_conn)))
                # VTK_TRIANGLE = 5
                elseif vtk_type == 5
                    push!(ferrite_cells, Ferrite.Triangle(Tuple(cell_conn)))
                # VTK_QUAD = 9
                elseif vtk_type == 9
                    push!(ferrite_cells, Ferrite.Quadrilateral(Tuple(cell_conn)))
                # VTK_LINE = 3
                elseif vtk_type == 3
                    push!(ferrite_cells, Ferrite.Line(Tuple(cell_conn)))
                else
                    @warn "Unsupported VTK cell type: $vtk_type, skipping"
                end
            end
            
            # Create Ferrite grid from nodes and cells
            grid = Ferrite.Grid(ferrite_cells, nodes)
            
            # Try to import cell data if present
            try
                cell_data = ReadVTK.get_cell_data(vtk_file)
                
                # Look for potential cell sets in cell data
                # Common names for cell entity IDs in different formats
                potential_cellset_names = ["CellEntityIds", "element_ids", "gmsh:physical", "ElementId"]
                
                for name in potential_cellset_names
                    if name in ReadVTK.keys(cell_data)
                        entity_ids_array = ReadVTK.get_data(cell_data[name])
                        unique_ids = unique(entity_ids_array)
                        
                        for id in unique_ids
                            cells_in_set = findall(entity_ids_array .== id)
                            if !isempty(cells_in_set)
                                Ferrite.addcellset!(grid, "cellset_$id", Set(cells_in_set))
                                println("  Added cellset_$id with $(length(cells_in_set)) cells")
                            end
                        end
                        
                        # Break after finding and processing the first valid cell set data
                        break
                    end
                end
            catch e
                @warn "Could not import cell data: $e"
            end
            
        else
            error("Unsupported mesh format: $ext. Only .msh and .vtu formats are supported.")
        end
        
        println("Mesh imported successfully: $(Ferrite.getnnodes(grid)) nodes, $(Ferrite.getncells(grid)) elements")
        
        return grid
    end

    import_mesh("data/cantilever_beam_volume_mesh.vtu")