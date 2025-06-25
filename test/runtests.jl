using Test
using LinearAlgebra  # Add this import for eigvals function
using TopOptEval
using TopOptEval.Utils

# include("VolumeForces/testVolumeForces.jl")


@testset "TopOptEval.jl" begin
    # Chapadlo test configuration flags
    RUN_raw_chapadlo = false
    RUN_lin_chapadlo = false
    RUN_sdf_chapadlo = true
    
    # Raw results from SIMP method (density field) - chapadlo version
    if RUN_raw_chapadlo
        @testset "Raw_chapadlo" begin
            # Task configuration
            taskName = "chapadlo_raw"
            
            # 1. Import mesh
            grid = import_mesh("../data/chapadlo/chapadlo-input_data.vtu")
            
            # 2. Extract density data
            density_data = extract_cell_density("../data/chapadlo/chapadlo-input_data.vtu")
            volume = Utils.calculate_volume(grid, density_data)
            
            # 3. Setup material model (SIMP) - chapadlo parameters
            # Material properties for Chapadlo
            E0 = 2.4e3      # MPa = N/mm²
            nu = 0.35       # Poisson's ratio
            Emin = 1e-8     # Minimum Young's modulus
            p = 3.0         # Penalization power
            
            # Create SIMP material model
            material_model = create_simp_material_model(E0, nu, Emin, p)
            
            # 4. Setup problem (initialize dof handler, cell values, matrices)
            dh, cellvalues, K, f = setup_problem(grid)
            
            # 5. Assemble stiffness matrix with variable material properties
            assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, density_data)
            
            # 6. Apply boundary conditions - chapadlo specific
            # Fixed support - circle on face (vetknutí)
            fixed_nodes = select_nodes_by_circle(grid, [0.0, 75.0, 115.0], [0.0, -1.0, 0.0], 16.11, 1e-3)
            
            # Symmetry - YZ plane at x = 0
            symmetry_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)
            
            # Force application points
            # Nožičky: plane at z = -90
            nozicky_nodes = select_nodes_by_plane(grid, [0.0, 0.0, -90.0], [0.0, 0.0, 1.0], 1.0)
            
            # Kamera: circular region at z = 5
            kamera_nodes = select_nodes_by_circle(grid, [0.0, 0.0, 5.0], [0.0, 0.0, 1.0], 21.5, 1e-3)
            
            # Check if sets are correct:
            all_force_nodes = union(nozicky_nodes, kamera_nodes)
            all_constraint_nodes = union(fixed_nodes, symmetry_nodes)
            export_boundary_conditions(grid, dh, all_constraint_nodes, all_force_nodes, "$(taskName)_boundary_conditions")
      exit()
            # 7. Apply boundary conditions
            # Fixed boundary condition (all DOFs)
            ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)
            
            # Symmetry boundary condition (X direction only)
            ch2 = apply_sliding_boundary!(K, f, dh, symmetry_nodes, [1])
            
            # Apply forces
            # Nožičky: 13N downward force
            apply_force!(f, dh, collect(nozicky_nodes), [0.0, 0.0, -13000.0]) # mN
            
            # Kamera: 0.5N downward force
            apply_force!(f, dh, collect(kamera_nodes), [0.0, 0.0, -500.0]) # mN
            
            # Apply acceleration: 6 m/s² in Y direction
            ρ = 1.04e-6  # kg/mm³
            apply_acceleration!(f, dh, cellvalues, [0.0, 6000.0, 0.0], ρ)
            
            # 8. Solve the system
            u, energy, stress_field, max_von_mises, max_stress_cell = solve_system_simp(K, f, dh, cellvalues, material_model, density_data, ch1, ch2)
    
            # 9. Print deformation energy and maximum stress
            @info "Final deformation energy: $energy J"
            @info "Deformation energy density: $(energy/(volume*10^(-9))) J/m^3"
            @info "Maximum von Mises stress: $max_von_mises at cell $max_stress_cell"
               
            # 10. Export results to ParaView
            export_results(u, dh, "$(taskName)_u")
            export_results(stress_field, dh, "$(taskName)_stress")
        end
    end
    
    # Results from linear post-processing - chapadlo version
    if RUN_lin_chapadlo
        @testset "Linear_chapadlo" begin
            # Task configuration
            taskName = "chapadlo_linear"
            
            # 1. Import mesh (tetrahedral mesh)
            grid = import_mesh("../data/chapadlo/chapadlo_B-2.0-nodal_STL.vtu")
            volume = calculate_volume(grid)
            
            # 2. Setup material model - chapadlo parameters
            E0 = 2.4e3      # MPa = N/mm²
            nu = 0.35       # Poisson's ratio
            λ, μ = create_material_model(E0, nu)
            
            # 3. Setup problem (initialize dof handler, cell values, matrices)
            dh, cellvalues, K, f = setup_problem(grid)
            
            # 4. Assemble stiffness matrix
            assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)
            
            # 5. Apply boundary conditions - chapadlo specific
            # Fixed support - circle on face (vetknutí)
            fixed_nodes = select_nodes_by_circle(grid, [0.0, 75.0, 115.0], [0.0, -1.0, 0.0], 16.11, 1e-3)
            
            # Symmetry - YZ plane at x = 0
            symmetry_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3)
            
            # Force application points
            # Nožičky: plane at z = -90
            nozicky_nodes = select_nodes_by_plane(grid, [0.0, 0.0, -90.0], [0.0, 0.0, 1.0], 1.0)
            
            # Kamera: circular region at z = 5
            kamera_nodes = select_nodes_by_circle(grid, [0.0, 0.0, 5.0], [0.0, 0.0, 1.0], 21.5, 1e-3)
    
            # Check if sets are correct:
            all_force_nodes = union(nozicky_nodes, kamera_nodes)
            all_constraint_nodes = union(fixed_nodes, symmetry_nodes)
            export_boundary_conditions(grid, dh, all_constraint_nodes, all_force_nodes, "$(taskName)_boundary_conditions")
            
      # exit()
            # 6. Apply boundary conditions
            # Fixed boundary condition (all DOFs)
            ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)
            
            # Symmetry boundary condition (X direction only)
            ch2 = apply_sliding_boundary!(K, f, dh, symmetry_nodes, [1])
            
            # Apply forces
            # Nožičky: 13N downward force
            apply_force!(f, dh, collect(nozicky_nodes), [0.0, 0.0, -13000.0]) # mN
            
            # Kamera: 0.5N downward force
            apply_force!(f, dh, collect(kamera_nodes), [0.0, 0.0, -500.0]) # mN
            
            # Apply acceleration: 6 m/s² in Y direction
            ρ = 1.04e-6  # kg/mm³
            apply_acceleration!(f, dh, cellvalues, [0.0, 6000.0, 0.0], ρ)
            
            # 7. Solve the system
            u, energy, stress_field, max_von_mises, max_stress_cell = solve_system(K, f, dh, cellvalues, λ, μ, ch1, ch2)
    
            # 8. Print deformation energy and maximum stress
            @info "Final deformation energy: $energy J"
            @info "Deformation energy density: $(energy/(volume*10^(-9))) J/m^3"
            @info "Maximum von Mises stress: $max_von_mises at cell $max_stress_cell"
            
            # 9. Export results to ParaView
            export_results(u, dh, "$(taskName)_u")
            export_results(stress_field, dh, "$(taskName)_stress")
        end
    end
    
    # Results from SDF post-processing - chapadlo version
    if RUN_sdf_chapadlo
        @testset "SDF_chapadlo" begin
            # Task configuration
            taskName = "chapadlo_sdf"
            
            # 1. Import mesh (tetrahedral mesh)
            grid = import_mesh("../data/chapadlo/tet_chapadlo_B-2.0_TriMesh-A15_cut.vtu")
            volume = Utils.calculate_volume(grid)
            
            # 2. Setup material model - chapadlo parameters
            E0 = 2.4e3      # MPa = N/mm²
            nu = 0.35       # Poisson's ratio
            λ, μ = create_material_model(E0, nu)
            
            # 3. Setup problem (initialize dof handler, cell values, matrices)
            dh, cellvalues, K, f = setup_problem(grid)
            
            # 5. Apply boundary conditions - chapadlo specific
            # Fixed support - circle on face (vetknutí)
            fixed_nodes = select_nodes_by_circle(grid, [0.0, 75.0, 115.0], [0.0, -1.0, 0.0], 16.11, 1e-3) # ok
            
            # Symmetry - YZ plane at x = 0
            symmetry_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-3) # ok
            
            # Force application points
            # Nožičky: plane at z = -90
            nozicky_nodes = select_nodes_by_plane(grid, [0.0, 0.0, -90.0], [0.0, 0.0, 1.0], 1.0) # ok
            
            # Kamera: circular region at z = 5
            kamera_nodes = select_nodes_by_circle(grid, [0.0, 0.0, 5.02], [0.0, 0.0, 1.0], 21.5, 1e-1) # ok
    
            # Check if sets are correct:
            all_force_nodes = union(nozicky_nodes, kamera_nodes)
            all_constraint_nodes = union(fixed_nodes, symmetry_nodes)
            export_boundary_conditions(grid, dh, all_constraint_nodes, all_force_nodes, "$(taskName)_boundary_conditions")
    
      # exit()
            # 4. Assemble stiffness matrix
            assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)

            # 6. Apply boundary conditions
            # Fixed boundary condition (all DOFs)
            ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)
            
            # Symmetry boundary condition (X direction only)
            ch2 = apply_sliding_boundary!(K, f, dh, symmetry_nodes, [1])
            
            # Apply forces
            # Nožičky: 13N downward force
            apply_force!(f, dh, collect(nozicky_nodes), [0.0, 0.0, -13000.0]) # mN
            
            # Kamera: 0.5N downward force
            apply_force!(f, dh, collect(kamera_nodes), [0.0, 0.0, -500.0]) # mN
            
            # Apply acceleration: 6 m/s² in Y direction
            ρ = 1.04e-6  # kg/mm³
            apply_acceleration!(f, dh, cellvalues, [0.0, 6000.0, 0.0], ρ)
             
            # 7. Solve the system
            u, energy, stress_field, max_von_mises, max_stress_cell = solve_system(K, f, dh, cellvalues, λ, μ, ch1, ch2)
    
            # 8. Print deformation energy and maximum stress
            @info "Final deformation energy: $energy J"
            @info "Deformation energy density: $(energy/(volume*10^(-9))) J/m^3"
            @info "Maximum von Mises stress: $max_von_mises at cell $max_stress_cell"
                       
            # 9. Export results to ParaView
            export_results(u, dh, "$(taskName)_u")
            export_results(stress_field, dh, "$(taskName)_stress")
        end
    end
end
