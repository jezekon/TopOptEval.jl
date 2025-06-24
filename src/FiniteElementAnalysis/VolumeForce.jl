"""
    apply_volume_force!(f, dh, cellvalues, body_force_vector, density=1.0)

Applies volume forces (body forces) such as gravity or acceleration to all elements in the mesh.
The volume force is integrated over each element and added to the global load vector.

Parameters:
- `f`: global load vector (modified in-place)
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `body_force_vector`: Body force per unit volume [Fx, Fy, Fz] in N/m³
- `density`: Material density in kg/m³ (default: 1.0)

Mathematical formulation:
f_body = ∫_Ω ρ * b * N dΩ

where:
- ρ is the density
- b is the body force per unit mass 
- N are the shape functions
- Ω is the element domain

Returns:
- nothing (modifies f in-place)
"""
function apply_volume_force!(f, dh, cellvalues, body_force_vector, density=1.0)
    # Convert body force to per unit mass if given per unit volume
    # If body_force_vector is already per unit mass (like gravity), use density=1.0
    body_force_per_mass = body_force_vector ./ density
    
    # Number of basis functions per element
    n_basefuncs = getnbasefunctions(cellvalues)
    
    # Element load vector for body forces
    fe_body = zeros(n_basefuncs)
    
    # Track total applied force for reporting
    total_force_applied = zeros(3)
    total_volume = 0.0
    
    # Iterate over all cells to apply volume forces
    for cell in CellIterator(dh)
        # Reinitialize cell values for current cell
        reinit!(cellvalues, cell)
        
        # Reset element load vector
        fill!(fe_body, 0.0)
        
        # Get cell DOFs
        cell_dofs = celldofs(cell)
        
        # Integrate body force over the element volume
        for q_point in 1:getnquadpoints(cellvalues)
            # Get integration weight and volume element
            dΩ = getdetJdV(cellvalues, q_point)
            total_volume += dΩ
            
            # Loop over all basis functions (DOFs in the element)
            for i in 1:n_basefuncs
                # Get shape function value at this quadrature point
                # For vector interpolation, this returns a Vec{3, Float64}
                N_vec = shape_value(cellvalues, q_point, i)
                
                # Calculate which spatial component this DOF corresponds to
                # For vector interpolation: DOF layout is [u1_x, u1_y, u1_z, u2_x, u2_y, u2_z, ...]
                dofs_per_node = 3  # 3D problem
                node_idx = div(i - 1, dofs_per_node) + 1
                dof_component = mod(i - 1, dofs_per_node) + 1
                
                # Extract the scalar shape function value for this component
                # The shape function vector has non-zero value only for the corresponding component
                N_scalar = N_vec[dof_component]
                
                # Add body force contribution: ρ * b * N * dΩ
                body_force_contribution = density * body_force_per_mass[dof_component] * N_scalar * dΩ
                fe_body[i] += body_force_contribution
                
                # Track total applied force
                total_force_applied[dof_component] += body_force_contribution
            end
        end
        
        # Add element contributions to global load vector
        for (local_dof, global_dof) in enumerate(cell_dofs)
            f[global_dof] += fe_body[local_dof]
        end
    end
    
    println("Applied volume force: $body_force_vector N/m³")
    println("Total force applied: $total_force_applied N")
    println("Total volume: $total_volume m³")
    println("Average force density: $(total_force_applied ./ total_volume) N/m³")
end

"""
    apply_gravity!(f, dh, cellvalues, density=1.0, g=9.81, direction=[0.0, 0.0, -1.0])

Convenience function to apply gravitational acceleration as a volume force.

Parameters:
- `f`: global load vector (modified in-place)
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `density`: Material density in kg/m³ (default: 1.0)
- `g`: Gravitational acceleration magnitude in m/s² (default: 9.81)
- `direction`: Gravity direction vector [x, y, z] (default: [0, 0, -1])

Returns:
- nothing (modifies f in-place)
"""
function apply_gravity!(f, dh, cellvalues, density=1.0, g=9.81, direction=[0.0, 0.0, -1.0])
    # Normalize direction vector
    unit_direction = direction ./ norm(direction)
    
    # Calculate gravitational force per unit volume
    gravity_force = density * g .* unit_direction
    
    println("Applying gravity: g = $g m/s², direction = $unit_direction, density = $density kg/m³")
    
    # Apply as volume force
    apply_volume_force!(f, dh, cellvalues, gravity_force, 1.0)  # density already included
end

"""
    apply_acceleration!(f, dh, cellvalues, acceleration_vector, density=1.0)

Applies a uniform acceleration field as a volume force (e.g., for dynamic analysis).

Parameters:
- `f`: global load vector (modified in-place)
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `acceleration_vector`: Acceleration vector [ax, ay, az] in m/s²
- `density`: Material density in kg/m³ (default: 1.0)

Returns:
- nothing (modifies f in-place)
"""
function apply_acceleration!(f, dh, cellvalues, acceleration_vector, density=1.0)
    # Calculate inertial force per unit volume: F = ρ * a
    inertial_force = density .* acceleration_vector
    
    println("Applying acceleration: a = $acceleration_vector m/s², density = $density kg/m³")
    
    # Apply as volume force
    apply_volume_force!(f, dh, cellvalues, inertial_force, 1.0)  # density already included
end

"""
    apply_variable_density_volume_force!(f, dh, cellvalues, body_force_vector, density_data)

Applies volume forces with variable density distribution (for SIMP topology optimization).

Parameters:
- `f`: global load vector (modified in-place)
- `dh`: DofHandler
- `cellvalues`: CellValues for interpolation and integration
- `body_force_vector`: Body force per unit mass [Fx, Fy, Fz] in N/kg
- `density_data`: Vector with density values for each cell

Returns:
- nothing (modifies f in-place)
"""
function apply_variable_density_volume_force!(f, dh, cellvalues, body_force_vector, density_data)
    # Number of basis functions per element
    n_basefuncs = getnbasefunctions(cellvalues)
    
    # Element load vector for body forces
    fe_body = zeros(n_basefuncs)
    
    # Track total applied force
    total_force_applied = zeros(3)
    
    # Iterate over all cells
    for cell in CellIterator(dh)
        # Get cell ID and corresponding density
        cell_id = cellid(cell)
        density = density_data[cell_id]
        
        # Skip if density is negligible (for SIMP optimization)
        if density < 1e-6
            continue
        end
        
        # Reinitialize cell values
        reinit!(cellvalues, cell)
        fill!(fe_body, 0.0)
        
        # Get cell DOFs
        cell_dofs = celldofs(cell)
        
        # Integrate body force over element volume
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            
            for i in 1:n_basefuncs
                # Get vector shape function value
                N_vec = shape_value(cellvalues, q_point, i)
                
                # Calculate DOF component
                dofs_per_node = 3
                dof_component = mod(i - 1, dofs_per_node) + 1
                
                # Extract scalar component
                N_scalar = N_vec[dof_component]
                
                # Apply variable density body force
                body_force_contribution = density * body_force_vector[dof_component] * N_scalar * dΩ
                fe_body[i] += body_force_contribution
                
                # Track total force
                total_force_applied[dof_component] += body_force_contribution
            end
        end
        
        # Add to global load vector
        for (local_dof, global_dof) in enumerate(cell_dofs)
            f[global_dof] += fe_body[local_dof]
        end
    end
    
    println("Applied variable density volume force")
    println("Total force applied: $total_force_applied N")
end
