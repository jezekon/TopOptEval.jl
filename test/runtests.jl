using Test
using LinearAlgebra  # Add this import for eigvals function
using TopOptEval
using TopOptEval.Utils

@testset "TopOptEval.jl" begin
    # Test configuration flags
    RUN_raw_beam = true
    RUN_lin_beam = true
    RUN_sdf_beam_approx = true
    RUN_sdf_beam_interp = true

    # Raw results from SIMP method (density field)
    if RUN_raw_beam
        @testset "Raw_beam" begin
            # Task configuration
            taskName = "beam_vfrac_04_Raw"
            
            # 1. Import mesh
            grid = import_mesh("../data/$(taskName).vtu")
            
            # 2. Extract density data
            density_data = extract_cell_density("../data/$(taskName).vtu")
            volume = Utils.calculate_volume(grid, density_data)
            
            # 3. Setup material model (SIMP)
            # Base properties
            E0 = 1.0  # Base Young's modulus
            nu = 0.3  # Poisson's ratio
            Emin = 1e-8  # Minimum Young's modulus
            p = 3.0  # Penalization power
            
            # Create SIMP material model
            material_model = create_simp_material_model(E0, nu, Emin, p)
            
            # 4. Setup problem (initialize dof handler, cell values, matrices)
            dh, cellvalues, K, f = setup_problem(grid)
            
            # 5. Assemble stiffness matrix with variable material properties
            assemble_stiffness_matrix_simp!(K, f, dh, cellvalues, material_model, density_data)
            
            # 6. Apply boundary conditions
            fixed_nodes = select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            # force_nodes = select_nodes_by_plane(grid, [60.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
            
            force_nodes = select_nodes_by_circle(
                grid,
                [60.0, 0.0, 2.0],  # center point
                [1.0, 0.0, 0.0],   # normal direction (perpendicular to the face)
                4.0                # radius of 5mm
            )
            # Check if sets are correct:
            export_boundary_conditions(grid, dh, fixed_nodes, force_nodes, "$(taskName)_boundary_conditions")
            
            # 7. Apply boundary conditions
            ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)
            
            # Apply 1N force in x direction to selected nodes
            apply_force!(f, dh, collect(force_nodes), [1.0, 0.0, 0.0])
            
            # 8. Solve the system
            u, energy, stress_field, max_von_mises, max_stress_cell = solve_system_simp(K, f, dh, cellvalues, material_model, density_data, ch1)
    
            # 9. Print deformation energy and maximum stress
            @info "Final deformation energy: $energy J"
            @info "Defromation energy density: $(energy/(volume*10^(-9))) J/m^3"
            @info "Maximum von Mises stress: $max_von_mises at cell $max_stress_cell"
               
            # 10. Export results to ParaView
            export_results(u, dh, "$(taskName)_u")
            export_results(stress_field, dh, "$(taskName)_stress")
        end
    end

    # Results from "linear" post-processing (extracted surface tri mesh from raw results + TetGen for volume (tet) mesh)
    if RUN_lin_beam
        @testset "Linear_beam" begin
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
            
            # 7. Solve the system
            u, energy, stress_field, max_von_mises, max_stress_cell = solve_system(K, f, dh, cellvalues, λ, μ, ch1)
    
            # 8. Print deformation energy and maximum stress
            @info "Final deformation energy: $energy J"
            @info "Defromation energy density: $(energy/(volume*10^(-9))) J/m^3"
            @info "Maximum von Mises stress: $max_von_mises at cell $max_stress_cell"
            
            # 9. Export results to ParaView
            export_results(u, dh, "$(taskName)_u")
            export_results(stress_field, dh, "$(taskName)_stress")
        end
    end
    
    # Results from SDF post-processing (tet mesh)
    if RUN_sdf_beam_approx
        @testset "SDF_beam_approx" begin
            # Task configuration

            taskName = "beam_vfrac_04_smooth-1_Approx"
            
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
            force_nodes = select_nodes_by_plane(grid, [60.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
            
            # Check if sets are correct:
            export_boundary_conditions(grid, dh, fixed_nodes, force_nodes, "$(taskName)_boundary_conditions")

            # 6. Apply boundary conditions
            ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)
            
            # Apply 1N force in x direction to selected nodes
            apply_force!(f, dh, collect(force_nodes), [1.0, 0.0, 0.0])
             
            # 7. Solve the system
            u, energy, stress_field, max_von_mises, max_stress_cell = solve_system(K, f, dh, cellvalues, λ, μ, ch1)
    
            # 8. Print deformation energy and maximum stress
            @info "Final deformation energy: $energy J"
            @info "Defromation energy density: $(energy/(volume*10^(-9))) J/m^3"
            @info "Maximum von Mises stress: $max_von_mises at cell $max_stress_cell"
            
            # 9. Export results to ParaView
            export_results(u, dh, "$(taskName)_u")
            export_results(stress_field, dh, "$(taskName)_stress")

        end
    end
   
    # Results from SDF post-processing (tet mesh)
    if RUN_sdf_beam_interp
        @testset "SDF_beam_interp" begin
            # Task configuration
            taskName = "beam_vfrac_04_smooth-1_Interp"
            
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
             
            # 7. Solve the system
            u, energy, stress_field, max_von_mises, max_stress_cell = solve_system(K, f, dh, cellvalues, λ, μ, ch1)
    
            # 8. Print deformation energy and maximum stress
            @info "Final deformation energy: $energy J"
            @info "Defromation energy density: $(energy/(volume*10^(-9))) J/m^3"
            @info "Maximum von Mises stress: $max_von_mises at cell $max_stress_cell"
                       
            # 9. Export results to ParaView
            export_results(u, dh, "$(taskName)_u")
            export_results(stress_field, dh, "$(taskName)_stress")

        end
    end
end
