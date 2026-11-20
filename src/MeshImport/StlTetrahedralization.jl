# Tetrahedral mesh of the volume inside a closed STL surface, made by TetGen through TetGen.jl.
#
# The triangles go to TetGen as the program `tetgen` reads them from an STL file: three points per triangle, in the
# order of the file, without merging the shared points. TetGen merges them itself. The mesh is then the same, point
# by point and tetrahedron by tetrahedron, as the output of the program `tetgen` of the same version with the same
# switches. A mesh made from merged points can differ, because TetGen inserts the points in another order.

"""
    read_stl_triangles(stl_file::String) -> Matrix{Float64}

Points of the triangles of an STL file, three columns per triangle, in the order of the file. Shared points are not
merged.
1. Format: binary when the file size is 84 + 50 n bytes, with n the number of triangles in bytes 81 to 84.
   Otherwise ASCII.
2. Binary: the three points of every triangle, single precision converted exactly to double precision. The normal
   and the attribute bytes are skipped.
3. ASCII: the three numbers of every line that starts with `vertex`.

TetGen decides the format by the first line: ASCII when it starts with "solid". The two rules agree, except for a
binary file whose header starts with "solid". TetGen cannot read such a file, this function can.

# Arguments
- `stl_file`: path to the STL file

# Returns
- matrix 3 × 3n with the points of the n triangles
"""
function read_stl_triangles(stl_file::String)
    # 1. Format from the file size
    file_size = filesize(stl_file)
    binary_count =
        file_size >= 84 ? open(io -> (skip(io, 80); Int(ltoh(read(io, UInt32)))), stl_file) : -1
    binary = file_size == 84 + 50 * binary_count

    # 2. Binary format
    if binary
        points = Matrix{Float64}(undef, 3, 3 * binary_count)
        open(stl_file) do io
            skip(io, 84)
            for triangle = 1:binary_count
                skip(io, 12)
                for corner = 1:3, axis = 1:3
                    points[axis, 3*(triangle-1)+corner] = Float64(ltoh(read(io, Float32)))
                end
                skip(io, 2)
            end
        end
        return points
    end

    # 3. ASCII format
    coordinates = Float64[]
    for line in eachline(stl_file)
        words = split(line)
        if !isempty(words) && words[1] == "vertex"
            append!(coordinates, parse.(Float64, words[2:4]))
        end
    end
    if isempty(coordinates) || length(coordinates) % 9 != 0
        error("No triangles read from $stl_file")
    end
    return reshape(coordinates, 3, :)
end

"""
    tetrahedralize_stl(stl_file::String; switches::String = "pYQ", hole_points = Vec{3,Float64}[]) -> Grid

Tetrahedral mesh of the volume inside the closed surface of an STL file, made by TetGen.
1. Read the triangles (`read_stl_triangles`).
2. Input of TetGen: the points, one facet per triangle, and the hole points.
3. Tetrahedralization with the switches.
4. Ferrite grid of linear tetrahedra, with the points and the node order of TetGen.

The default switches `pYQ` mesh the closed surface (`p`, a piecewise linear complex), keep the triangles of the
surface as they are (`Y`) and silence TetGen (`Q`). The surface of the mesh is then the surface of the STL file. Without
`Y`, TetGen joins adjacent triangles that lie nearly in one plane and splits them again along other diagonals, so the
points stay but some triangles of the surface change. Other switches are those of the program `tetgen`, for example
`q` for a quality mesh. Without hole points, TetGen fills a closed cavity with tetrahedra. One point inside the cavity
keeps it empty.

# Arguments
- `stl_file`: path to the STL file with a closed triangle surface
- `switches`: switches of TetGen, as for the program `tetgen`
- `hole_points`: one point inside each closed cavity that stays empty

# Returns
- `Ferrite.Grid` with `Tetrahedron` cells
"""
function tetrahedralize_stl(
    stl_file::String;
    switches::String = "pYQ",
    hole_points = Vec{3,Float64}[],
)
    # 1. Triangles of the STL file
    println("Tetrahedralizing $stl_file with TetGen (switches $switches)...")
    points = read_stl_triangles(stl_file)

    # 2. Input of TetGen: point k of the file is point k of the input, facet t holds points 3t-2, 3t-1, 3t
    input = TetGen.RawTetGenIO{Float64}()
    input.pointlist = points
    TetGen.facetlist!(input, reshape(Cint.(1:size(points, 2)), 3, :))
    if !isempty(hole_points)
        input.holelist = reduce(hcat, [Float64[point[1], point[2], point[3]] for point in hole_points])
    end

    # 3. Tetrahedralization
    output = TetGen.tetrahedralize(input, switches)

    # 4. Ferrite grid
    nodes = [Ferrite.Node(Vec{3}(Tuple(output.pointlist[:, i]))) for i in axes(output.pointlist, 2)]
    cells = [
        Ferrite.Tetrahedron(Tuple(Int.(output.tetrahedronlist[:, k]))) for
        k in axes(output.tetrahedronlist, 2)
    ]
    grid = Ferrite.Grid(cells, nodes)
    println("Mesh created: $(Ferrite.getnnodes(grid)) nodes, $(Ferrite.getncells(grid)) elements")
    return grid
end
