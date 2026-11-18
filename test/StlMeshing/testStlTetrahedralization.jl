# Tests of read_stl_triangles and tetrahedralize_stl on boxes written as STL files in a temporary folder.
# The corners have the coordinates 0, 0.5, 1.5 and 2. Single precision holds them exactly, so the binary and the
# ASCII file give the same points. The tetrahedra of a box have exact volumes up to the rounding of a sum, so the
# volume of a mesh equals the volume of the box to 1e-12.
using Ferrite
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
    end
end
