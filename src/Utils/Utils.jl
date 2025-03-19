module Utils

export calculate_volume

using Ferrite

"""
    calculate_volume(grid::Ferrite.Grid)

Calculates the total volume of a mesh consisting of tetrahedral elements.

Parameters:
- `grid`: Ferrite Grid object representing the mesh

Returns:
- The total volume of the mesh in cubic units
"""
function calculate_volume(grid::Ferrite.Grid)
    # Initialize total volume
    total_volume = 0.0
    
    # Create a scalar interpolation (sufficient for volume integration)
    ip = Lagrange{RefTetrahedron, 1}()
    
    # Create quadrature rule for integration (linear is sufficient for volume)
    qr = QuadratureRule{RefTetrahedron}(1)
    
    # Create cell values for integration
    cellvalues = CellValues(qr, ip)
    
    # Create DofHandler to iterate over cells
    dh = DofHandler(grid)
    add!(dh, :u, ip)  # Add a scalar field for volume calculation
    close!(dh)
    
    # Iterate over all cells
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        
        # For each quadrature point, add the integration weight times Jacobian determinant
        cell_volume = 0.0
        for q_point in 1:getnquadpoints(cellvalues)
            # getdetJdV provides the volume element at the quadrature point
            dΩ = getdetJdV(cellvalues, q_point)
            cell_volume += dΩ
        end
        
        total_volume += cell_volume
    end
    
    @info "Total mesh volume: $total_volume cubic units"
    return total_volume
end


end
