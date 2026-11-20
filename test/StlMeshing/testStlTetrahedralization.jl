# Tests of read_stl_triangles and tetrahedralize_stl on boxes written as STL files in a temporary folder.
# The corners have the coordinates 0, 0.5, 1.5 and 2. Single precision holds them exactly, so the binary and the
# ASCII file give the same points. The tetrahedra of a box have exact volumes up to the rounding of a sum, so the
# volume of a mesh equals the volume of the box to 1e-12.
using Ferrite
using LinearAlgebra
using TopOptEval
using TopOptEval.Utils

"""
    box_triangles(low, high) -> Vector{NTuple{3,NTuple{3,Float64}}}

The 12 triangles of the surface of the box [low, high]^3, two per face.
"""
function box_triangles(low, high)
    corner(i, j, k) = (i == 0 ? low : high, j == 0 ? low : high, k == 0 ? low : high)
    faces = [
        ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),    # z = low
        ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),    # z = high
        ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),    # y = low
        ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),    # y = high
        ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),    # x = low
        ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),    # x = high
    ]
    triangles = NTuple{3,NTuple{3,Float64}}[]
    for (a, b, c, d) in faces
        push!(triangles, (corner(a...), corner(b...), corner(c...)))
        push!(triangles, (corner(a...), corner(c...), corner(d...)))
    end
    return triangles
end

"""
    write_binary_stl(file, triangles)

Binary STL: 80-byte header, the number of triangles, then per triangle a zero normal, three points and two
attribute bytes.
"""
function write_binary_stl(file, triangles)
    open(file, "w") do io
        write(io, zeros(UInt8, 80))
        write(io, UInt32(length(triangles)))
        for triangle in triangles
            write(io, zeros(Float32, 3))
            for point in triangle
                write(io, Float32.(collect(point)))
            end
            write(io, UInt16(0))
        end
    end
end

"""
    write_ascii_stl(file, triangles)

ASCII STL with a zero normal for every triangle.
"""
function write_ascii_stl(file, triangles)
    open(file, "w") do io
        println(io, "solid box")
        for triangle in triangles
            println(io, "  facet normal 0 0 0")
            println(io, "    outer loop")
            for point in triangle
                println(io, "      vertex $(point[1]) $(point[2]) $(point[3])")
            end
            println(io, "    endloop")
            println(io, "  endfacet")
        end
        println(io, "endsolid box")
    end
end

"""
    roof_box_triangles(lift, diagonal) -> Vector{NTuple{3,NTuple{3,Float64}}}

The 12 triangles of the box [0, 2] × [0, 1] × [0, 1] whose top corner (0, 1, 1) is lifted by `lift`. The top face
is split along the diagonal from (0, 0, 1) to (2, 1, 1) (`diagonal = 1`) or from (2, 0, 1) to (0, 1, 1 + lift)
(`diagonal = 2`). The other faces are as in `box_triangles`.
"""
function roof_box_triangles(lift, diagonal)
    corner(i, j, k) = (2.0 * i, Float64(j), k + (i == 0 && j == 1 && k == 1 ? lift : 0.0))
    triangles = [(map(p -> corner(round.(Int, p)...), t)) for t in box_triangles(0.0, 1.0)]
    a, b, c, d = corner(0, 0, 1), corner(1, 0, 1), corner(1, 1, 1), corner(0, 1, 1)
    top = diagonal == 1 ? [(a, b, c), (a, c, d)] : [(a, b, d), (b, c, d)]
    return [triangles[1:2]; top; triangles[5:end]]
end

"""
    triangle_set(points) -> Set

Triangles given by the columns of a 3 × 3n matrix (as from `read_stl_triangles`), each as its three points in
sorted order, so that the order of the points in the file does not matter.
"""
triangle_set(points) =
    Set(Tuple(sort([Tuple(points[:, t+k]) for k = 0:2])) for t = 1:3:size(points, 2))

"""
    boundary_set(grid) -> Set

Boundary faces of a tetrahedral grid, the faces of exactly one tetrahedron, in the form of `triangle_set`.
"""
function boundary_set(grid)
    count = Dict{Vector{Int},Int}()
    for cell in getcells(grid), face in ((1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4))
        nodes = sort([cell.nodes[k] for k in face])
        count[nodes] = get(count, nodes, 0) + 1
    end
    faces = [nodes for (nodes, n) in count if n == 1]
    return Set(Tuple(sort([Tuple(grid.nodes[i].x) for i in nodes])) for nodes in faces)
end

"""
    enclosed_volume(points) -> Float64

Volume inside a closed surface with outward triangles, by the divergence theorem: the sum of the signed volumes of
the tetrahedra from the origin to each triangle.
"""
function enclosed_volume(points)
    volume = 0.0
    for t = 1:3:size(points, 2)
        a, b, c = points[:, t], points[:, t+1], points[:, t+2]
        volume += dot(a, cross(b, c)) / 6
    end
    return volume
end

"""Runs `f` with its printed output discarded."""
silent(f) = redirect_stdout(f, devnull)

@testset "STL tetrahedralization" begin
    mktempdir() do folder
        cube = box_triangles(0.0, 2.0)
        cavity = box_triangles(0.5, 1.5)
        cube_binary = joinpath(folder, "cube_binary.stl")
        cube_ascii = joinpath(folder, "cube_ascii.stl")
        hollow_cube = joinpath(folder, "hollow_cube.stl")
        write_binary_stl(cube_binary, cube)
        write_ascii_stl(cube_ascii, cube)
        write_binary_stl(hollow_cube, [cube; cavity])

        # The 12 triangles give 36 points, three per triangle in the order of the file. The ASCII file holds the
        # same numbers, so both formats give the same matrix.
        @testset "Reading binary and ASCII STL" begin
            points = read_stl_triangles(cube_binary)
            @test size(points) == (3, 36)
            @test points[:, 4:6] == hcat(collect.(cube[2])...)
            @test read_stl_triangles(cube_ascii) == points
        end

        # TetGen merges the 36 points into the 8 corners. The switch p adds no point to a box, so the mesh has the
        # 8 corners as nodes and the volume 2^3 = 8.
        @testset "Cube" begin
            grid = silent(() -> tetrahedralize_stl(cube_binary))
            @test getnnodes(grid) == 8
            @test getncells(grid) > 0
            @test calculate_volume(grid) ≈ 8.0 atol = 1e-12
            @test silent(() -> getncells(tetrahedralize_stl(cube_ascii))) == getncells(grid)
        end

        # A cube 2 × 2 × 2 with a closed cavity 1 × 1 × 1. Without a hole point TetGen fills the cavity, so the
        # volume is 8. With the hole point (1, 1, 1) the cavity stays empty and the volume is 8 − 1 = 7.
        @testset "Cube with a cavity" begin
            filled = silent(() -> tetrahedralize_stl(hollow_cube))
            @test getnnodes(filled) == 16
            @test calculate_volume(filled) ≈ 8.0 atol = 1e-12
            hollow = silent(() -> tetrahedralize_stl(hollow_cube; hole_points = [Vec(1.0, 1.0, 1.0)]))
            @test getnnodes(hollow) == 16
            @test calculate_volume(hollow) ≈ 7.0 atol = 1e-12
        end

        # A box 2 × 1 × 1 whose top corner (0, 1, 1) is lifted by 1e-4, so the two triangles of the top face lie
        # nearly in one plane. The two diagonals of the top face give two surfaces with different volumes. The
        # default switch Y keeps the triangles of the file, so the boundary faces of the mesh are the triangles of
        # the file, and the volume of the mesh equals the volume inside the surface up to the rounding of a sum.
        # The ASCII file holds each number as Julia prints it, which reads back to the same value, so the points
        # read back are exact.
        @testset "Surface of the file kept, diagonal $diagonal" for diagonal in (1, 2)
            file = joinpath(folder, "roof_$diagonal.stl")
            write_ascii_stl(file, roof_box_triangles(1e-4, diagonal))
            points = read_stl_triangles(file)
            grid = silent(() -> tetrahedralize_stl(file))
            @test getnnodes(grid) == 8
            @test boundary_set(grid) == triangle_set(points)
            @test calculate_volume(grid) ≈ enclosed_volume(points) atol = 1e-12
        end
    end
end
