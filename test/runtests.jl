using Test
using LinearAlgebra  # Add this import for eigvals function
using TopOptEval
using TopOptEval.Utils

# include("VolumeForces/testVolumeForces.jl")

# INFO: Input data:
# 1_Gridap_geom-Raw_SIMP.vtu      2_Gridap_geom_linear.vtu        3_Gridap_geom_SDF.vtu           4_Gridap_geom_level-set.vtu

@testset "TopOptEval.jl" begin
    # Chapadlo test configuration flags

    RUN_raw_gridap_geom = true
    RUN_lin_gridap_geom = true
    RUN_sdf_gridap_geom = true
    RUN_level_gridap_geom = true

    # Raw results from SIMP method (density field) - chapadlo version
    if RUN_raw_gridap_geom
        @testset "RUN_raw_gridap_geom" begin
            # Task configuration
            taskName = "Gridap_geom_raw"

            # 1. Import mesh
            grid = import_mesh("../data/Gridap_geom/1_Gridap_geom-Raw_SIMP.vtu")

            # 2. Extract density data
            density_data =
                extract_cell_density("../data/Gridap_geom/1_Gridap_geom-Raw_SIMP.vtu")
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
            assemble_stiffness_matrix_simp!(
                K,
                f,
                dh,
                cellvalues,
                material_model,
                density_data,
            )

            # 6. Apply boundary conditions
            fixed_nodes =
                select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.01)
            # force_nodes = select_nodes_by_plane(grid, [60.0, 0.0, 0.0], [-1.0, 0.0, 0.0])

            force_nodes = select_nodes_by_circle(
                grid,
                [2.0, 0.5, 0.5],  # center pointtrue
                [-1.0, 0.0, 0.0],   # normal direction (perpendicular to the face)
                0.1 + eps(),                # radius of 5mm
                0.01,
            )
            # Check if sets are correct:
            export_boundary_conditions(
                grid,
                dh,
                fixed_nodes,
                force_nodes,
                "$(taskName)_boundary_conditions",
            )

            # 7. Apply boundary conditions
            ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)

            # Apply 1N force in x direction to selected nodes
            apply_force!(f, dh, collect(force_nodes), [0.0, 0.0, -1.0])

            # 8. Solve the system
            u, energy, stress_field, max_von_mises, max_stress_cell =
                solve_system_simp(K, f, dh, cellvalues, material_model, density_data, ch1)

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
    if RUN_lin_gridap_geom
        @testset "RUN_lin_gridap_geom" begin
            # Task configuration
            taskName = "Gridap_geom_linear"

            # 1. Import mesh (vtu/msh)
            grid = import_mesh("../data/Gridap_geom/2_Gridap_geom_linear.vtu")
            volume = calculate_volume(grid)

            # 2. Setup material model (steel)
            λ, μ = create_material_model(1.0, 0.3)

            # 3. Setup problem (initialize dof handler, cell values, matrices)
            dh, cellvalues, K, f = setup_problem(grid)

            # 4. Assemble stiffness matrix
            assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)

            # 5. Apply boundary conditions
            fixed_nodes =
                select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.01)
            # Force application on a circular region at x=2mm
            force_nodes = select_nodes_by_circle(
                grid,
                [2.0, 0.5, 0.5],  # center pointtrue
                [-1.0, 0.0, 0.0], # normal direction (perpendicular to the face)
                0.1 + eps(),      # radius of 0.1mm
                0.01,
            )

            # Check if sets are correct:
            export_boundary_conditions(
                grid,
                dh,
                fixed_nodes,
                force_nodes,
                "$(taskName)_boundary_conditions",
            )
            # exit()

            # 6. Apply boundary conditions
            ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)

            # Apply 1N force in x direction to selected nodes
            apply_force!(f, dh, collect(force_nodes), [0.0, 0.0, -1.0])

            # 7. Solve the system
            u, energy, stress_field, max_von_mises, max_stress_cell =
                solve_system(K, f, dh, cellvalues, λ, μ, ch1)

            # 8. Print deformation energy and maximum stress
            @info "Final deformation energy: $energy J"
            @info "Defromation energy density: $(energy/(volume*10^(-9))) J/m^3"
            @info "Maximum von Mises stress: $max_von_mises at cell $max_stress_cell"

            # 9. Export results to ParaView
            export_results(u, dh, "$(taskName)_u")
            export_results(stress_field, dh, "$(taskName)_stress")
        end
    end

    # Results from "sdf" post-processing (extracted surface tri mesh from raw results + TetGen for volume (tet) mesh)
    if RUN_sdf_gridap_geom
        @testset "RUN_sdf_gridap_geom" begin
            # Task configuration
            taskName = "Gridap_geom_sdf"

            # 1. Import mesh (vtu/msh)
            grid = import_mesh("../data/Gridap_geom/3_Gridap_geom_SDF.vtu")
            volume = calculate_volume(grid)

            # 2. Setup material model (steel)
            λ, μ = create_material_model(1.0, 0.3)

            # 3. Setup problem (initialize dof handler, cell values, matrices)
            dh, cellvalues, K, f = setup_problem(grid)

            # 4. Assemble stiffness matrix
            assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)

            # 5. Apply boundary conditions
            fixed_nodes =
                select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.01)
            # Force application on a circular region at x=2mm
            force_nodes = select_nodes_by_circle(
                grid,
                [2.0, 0.5, 0.5],  # center pointtrue
                [-1.0, 0.0, 0.0], # normal direction (perpendicular to the face)
                0.1 + eps(),      # radius of 0.1mm
                0.01,
            )

            # Check if sets are correct:
            export_boundary_conditions(
                grid,
                dh,
                fixed_nodes,
                force_nodes,
                "$(taskName)_boundary_conditions",
            )
            # exit()

            # 6. Apply boundary conditions
            ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)

            # Apply 1N force in x direction to selected nodes
            apply_force!(f, dh, collect(force_nodes), [0.0, 0.0, -1.0])

            # 7. Solve the system
            u, energy, stress_field, max_von_mises, max_stress_cell =
                solve_system(K, f, dh, cellvalues, λ, μ, ch1)

            # 8. Print deformation energy and maximum stress
            @info "Final deformation energy: $energy J"
            @info "Defromation energy density: $(energy/(volume*10^(-9))) J/m^3"
            @info "Maximum von Mises stress: $max_von_mises at cell $max_stress_cell"

            # 9. Export results to ParaView
            export_results(u, dh, "$(taskName)_u")
            export_results(stress_field, dh, "$(taskName)_stress")
        end
    end

    # Results from level-set (extracted surface tri mesh from raw results + TetGen for volume (tet) mesh)
    if RUN_level_gridap_geom
        @testset "RUN_level_gridap_geom" begin
            # Task configuration
            taskName = "Gridap_geom_level-set"

            # 1. Import mesh (vtu/msh)
            grid = import_mesh("../data/Gridap_geom/4_Gridap_geom_level-set.vtu")
            volume = calculate_volume(grid)

            # 2. Setup material model (steel)
            λ, μ = create_material_model(1.0, 0.3)

            # 3. Setup problem (initialize dof handler, cell values, matrices)
            dh, cellvalues, K, f = setup_problem(grid)

            # 4. Assemble stiffness matrix
            assemble_stiffness_matrix!(K, f, dh, cellvalues, λ, μ)

            # 5. Apply boundary conditions
            fixed_nodes =
                select_nodes_by_plane(grid, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.01)
            # Force application on a circular region at x=2mm
            force_nodes = select_nodes_by_circle(
                grid,
                [2.0, 0.5, 0.5],  # center pointtrue
                [-1.0, 0.0, 0.0], # normal direction (perpendicular to the face)
                0.1 + eps(),      # radius of 0.1mm
                0.01,
            )

            # Check if sets are correct:
            export_boundary_conditions(
                grid,
                dh,
                fixed_nodes,
                force_nodes,
                "$(taskName)_boundary_conditions",
            )

            # 6. Apply boundary conditions
            ch1 = apply_fixed_boundary!(K, f, dh, fixed_nodes)

            # Apply 1N force in x direction to selected nodes
            apply_force!(f, dh, collect(force_nodes), [0.0, 0.0, -1.0])

            # 7. Solve the system
            u, energy, stress_field, max_von_mises, max_stress_cell =
                solve_system(K, f, dh, cellvalues, λ, μ, ch1)

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
