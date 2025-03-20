using Test
using LinearAlgebra  # Add this import for eigvals function
using TopOptEval
using TopOptEval.Utils

@testset "TopOptEval.jl" begin
    # Test configuration flags
    RUN_lin_beam = false
    RUN_sdf_beam = false
    RUN_raw_beam = true
    
    if RUN_lin_beam
        @testset "Linear_beam" begin  # Added the missing 'begin' keyword here
            # Task configuration
            taskName = "beam_linear_volume_mesh"
            
            # 1. Import mesh (vtu/msh)
            grid = import_mesh("../data/$(taskName).vtu")
            # grid = import_mesh("../data/$(taskName).msh")
            volume = calculate_volume(grid)
            
            # 2. Setup material model (steel)
            λ, μ = create_material_model(1.0, 0.3)
            
            # 3. Setup problem (initialize dof handler, cell values, matrices)
            dh, cellvalues, K, f = setup_problem(grid)
            
            # 4. Assemble stiffness matrix
            assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)
            
            # 5. Apply boundary conditions
            fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            # Force application on a circular region at x=60mm
            force_nodes = select_nodes_by_circle(
                grid,
                [60.0, 0.0, 0.0],  # center point
                [1.0, 0.0, 0.0],   # normal direction (perpendicular to the face)
                5.0                # radius of 5mm
            )

            # Check if sets are correct:
            export_boundary_conditions(grid, dh, fixed_nodes, force_nodes, "$(taskName)_boundary_conditions")
            
            # 6. Apply boundary conditions
            ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)
            
            # Apply 1N force in x direction to selected nodes
            apply_force!(f, dh, collect(force_nodes), [1.0, 0.0, 0.0])
            
            # 7. Solve the system - using the new signature that includes stress calculation
            u, energy, stress_field = solve_system(K, f, dh, cellvalues, λ, μ, ch1) # ch2, ch3, ...
            
            # 8. Print deformation energy
            @info "Final deformation energy: $energy J"

            println(typeof(stress_field))
            
            # 9. Export results to ParaView
            export_results(u, dh, "$(taskName)_u")
            export_results(stress_field, dh, "$(taskName)_stress")
        end
    end
    
    # Other test cases can be added similarly
    if RUN_sdf_beam
        @testset "SDF_beam" begin
            # Task configuration
            taskName = "beam_vfrac_04_smooth-2_Approx"
            # taskName = "beam_vfrac_04_smooth-2_Interp"
            
            # 1. Import mesh (vtu/msh)
            grid = import_mesh("../data/$(taskName).vtu")
            volume = Utils.calculate_volume(grid)
            
            # 2. Setup material model (steel)
            λ, μ = create_material_model(1.0, 0.3)
            
            # 3. Setup problem (initialize dof handler, cell values, matrices)
            dh, cellvalues, K, f = setup_problem(grid)
            
            # 4. Assemble stiffness matrix
            assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)
            
            # 5. Apply boundary conditions
            fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            # Force application on a circular region at x=60mm
            # force_nodes = select_nodes_by_circle(
            #     grid,
            #     [60.0, 0.0, 0.0],  # center point
            #     [1.0, 0.0, 0.0],   # normal direction (perpendicular to the face)
            #     5.0                # radius of 5mm
            # )
            force_nodes = select_nodes_by_plane(grid, [60.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
            println(length(force_nodes))
            
            # Check if sets are correct:
            export_boundary_conditions(grid, dh, fixed_nodes, force_nodes, "$(taskName)_boundary_conditions")

            # 6. Apply boundary conditions
            ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)
            
            # Apply 1N force in x direction to selected nodes
            apply_force!(f, dh, collect(force_nodes), [1.0, 0.0, 0.0])
            
            # 7. Solve the system - using the new signature that includes stress calculation
            u, energy, stress_field = solve_system(K, f, dh, cellvalues, λ, μ, ch1) # ch2, ch3, ...
            
            # 8. Print deformation energy
            @info "Final deformation energy: $energy J"

            println(typeof(stress_field))
            
            # 9. Export results to ParaView
            export_results(u, dh, "$(taskName)_u")
            export_results(stress_field, dh, "$(taskName)_stress")

        end
    end
    
    if RUN_raw_beam
        @testset "Raw_beam" begin
            # Task configuration
            taskName = "beam_vfrac_04_Raw"
            
            # 1. Import mesh
            grid = import_mesh("../data/$(taskName).vtu")
            # volume = Utils.calculate_volume(grid)
            
            # 2. Extract density data
            density_data = MeshImport.extract_cell_density("../data/$(taskName).vtu")
            
            # 3. Setup material model (SIMP)
            # Base properties
            E0 = 1.0  # Base Young's modulus
            nu = 0.3  # Poisson's ratio
            Emin = 1e-9  # Minimum Young's modulus
            p = 3.0  # Penalization power
            
            # Create SIMP material model
            material_model = create_simp_material_model(E0, nu, Emin, p)
            
            # 4. Setup problem (initialize dof handler, cell values, matrices)
            dh, cellvalues, K, f = setup_problem(grid)
            
            # 5. Assemble stiffness matrix with variable material properties
            assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, density_data)
            
            # 6. Apply boundary conditions
            fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            force_nodes = select_nodes_by_plane(grid, [60.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
            
            # Check if sets are correct:
            export_boundary_conditions(grid, dh, fixed_nodes, force_nodes, "$(taskName)_boundary_conditions")
            
            # 7. Apply boundary conditions
            ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)
            
            # Apply 1N force in x direction to selected nodes
            apply_force!(f, dh, collect(force_nodes), [1.0, 0.0, 0.0])
            
            # 8. Solve the system
            u, energy, stress_field = solve_system_simp(K, f, dh, cellvalues, material_model, density_data, ch1)
            
            # 9. Print deformation energy
            @info "Final deformation energy: $energy J"
            
            # 10. Export results to ParaView
            export_results(u, dh, "$(taskName)_u")
            export_results(stress_field, dh, "$(taskName)_stress")
        end
    end
end
