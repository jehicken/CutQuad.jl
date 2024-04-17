module CutQuad

using RegionTrees, CxxWrap

using AlgoimDiff_jll
if !isdefined(AlgoimDiff_jll, :libcutquad_path)
    error("Sorry!  AlgoimDiff_jll is not available on this platform")
end
@wrapmodule(AlgoimDiff_jll.get_libcutquad_path)

# Following is for local testing of the cutquad library
#@wrapmodule(() -> joinpath("/home/jehicken/Libraries/algoim","libcutquad.so"))

__init__() = @initcxx

# Module variable used to store a reference to the levset;
# this is needed for the @safe_cfunction macro
const mod_levset = Ref{Any}()

"""
    wts, pts = cut_cell_quad(cell, phi, num_quad [, fit_degree=num_quad-1])

Returns a volume quadrature for the domain defined by the HyperRectangle `cell` 
and level-set function `phi`.  The canonical quadrature on a non-cut domain
(i.e. `phi(x) > 0` for all `x`) uses a tensor-product Gauss-Legendre quadrature 
with `num_quad^Dim` points total, where `Dim` is the dimension of the domain.
The `fit_degree` parameter indicates the degree of the Bernstein polynomial that
the algoim library uses to represent the level-set function `phi`.  

# Example
```julia-repl
julia> using CutQuad, RegionTrees

julia> using StaticArrays : SVector

julia> phi = x-> 4*(x[1] + 1)^2 + 36*(x[2] - 0.5)^2 - 9;

julia> cell = HyperRectangle(SVector{2}([0.0; 0.0]), SVector{2}([1.0; 1.0]));

julia> wts, pts = cut_cell_quad(cell, phi, 3, fit_degree=2)
```
"""
function cut_cell_quad(cell::HyperRectangle{1,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    pts = Vector{Float64}(); wts = Vector{Float64}()
    xwork = zeros(1) # allocating here has insignificant impact, it seems
    mod_levset[] = phi
    cphi = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))
    cut_cell_quad1d(cphi, num_quad, fit_degree,
                    Vector{Float64}(cell.origin),
                    Vector{Float64}(cell.origin + cell.widths),
                    xwork, wts, pts)
    return wts, reshape(pts, 1, :)
end

function cut_cell_quad(cell::HyperRectangle{2,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    pts = Vector{Float64}(); wts = Vector{Float64}()
    xwork = zeros(2) # allocating here has insignificant impact, it seems
    mod_levset[] = phi
    cphi = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))
    cut_cell_quad2d(cphi, num_quad, fit_degree,
                    Vector{Float64}(cell.origin),
                    Vector{Float64}(cell.origin + cell.widths),
                    xwork, wts, pts)
    return wts, reshape(pts, 2, :)
end

function cut_cell_quad(cell::HyperRectangle{3,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    pts = Vector{Float64}(); wts = Vector{Float64}()    
    xwork = zeros(3) # allocating here has insignificant impact, it seems
    mod_levset[] = phi
    cphi = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))
    cut_cell_quad3d(cphi, num_quad, fit_degree,
                    Vector{Float64}(cell.origin),
                    Vector{Float64}(cell.origin + cell.widths),
                    xwork, wts, pts)
    return wts, reshape(pts, 3, :)
end

"""
    wts, pts = cut_face_quad(face, dir, phi, num_quad [, fit_degree=num_quad-1])

Returns a quadrature for the face defined by the HyperRectangle `face` and 
level-set function `phi`.  The face has a unit normal whose only non-zero
component is in the direction `dir`; furthermore, if the spatial dimension is
`Dim` --- which is determined by the parameter in `face` --- the HyperRectangle
should have a width of zero in the direction `dir`.  The canonical quadrature
on a non-cut face (i.e. `phi(x) > 0` for all `x` over `face`) uses a
tensor-product Gauss-Legendre quadrature with `num_quad^(Dim-1)` points total.
The `fit_degree` parameter indicates the degree of the Bernstein polynomial
that the algoim library uses to represent the level-set function `phi`.  

# Example
```julia-repl
julia> using CutQuad, RegionTrees

julia> using StaticArrays : SVector

julia> phi = x-> 4*(x[1] + 1)^2 + 36*(x[2] - 0.5)^2 - 9;

julia> face = HyperRectangle(SVector{2}([0.0; 0.0]), SVector{2}([0.0; 1.0]));

julia> face_wts, face_pts = cut_face_quad(face, 1, phi, 3, fit_degree=2)
```
"""
function cut_face_quad(face::HyperRectangle{Dim,Float64}, dir::Int, phi, 
                       num_quad::Int; fit_degree::Int=num_quad-1) where {Dim}
    @assert( dir > 0 && dir <= Dim, "dir for face must be between 1 and Dim" )
    @assert( abs(face.widths[dir]) < eps(), "face.widths[dir] is not zero" )
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )

    # helper function for mapping dimension `Dim-1` indices to dimension `Dim`
    lift_index = i -> mod(dir+i-1,Dim)+1

    # define function that restricts level-set to slice x[dir] = xdir
    xdir = face.origin[dir]
    z = zeros(Dim)
    function face_phi(x)
        z[dir] = xdir
        for i = 1:Dim-1
            z[lift_index(i)] = x[i]
        end
        return phi(z)
    end
    mod_levset[] = face_phi
    cphi = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))

    # get bounding box for `Dim-1` dimensional cut 
    x_min = zeros(Dim-1)
    x_max = zeros(Dim-1)
    for i = 1:Dim-1
        x_min[i] = face.origin[lift_index(i)]
        x_max[i] = x_min[i] + face.widths[lift_index(i)]
    end

    # call appropriate algorim library function
    pts = Vector{Float64}(); wts = Vector{Float64}()
    xwork = zeros(Dim-1)
    if Dim == 2
        cut_cell_quad1d(cphi, num_quad, fit_degree, x_min, x_max, xwork, 
                        wts, pts)
    elseif Dim == 3
        cut_cell_quad2d(cphi, num_quad, fit_degree, x_min, x_max, xwork, 
                        wts, pts)
    end

    if length(wts) == 0
        return wts, zeros(Dim, 0)
    end

    # expand `Dim-1` dimensional points into `Dim` dimensional points
    pts2d = reshape(pts, Dim-1, :)
    pts_full = zeros(Dim, length(wts))
    pts_full[dir,:] .= xdir
    for i = 1:Dim-1
        pts_full[lift_index(i),:] .= pts2d[i,:]
    end
    return wts, pts_full
end

function cut_face_quad(face::HyperRectangle{1,Float64}, dir::Int, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    xdir = face.origin[dir]
    if phi(xdir) > 0.0
        pts = xdir*ones(1,1)
        wts = ones(1)
    else 
        pts = zeros(1,0)
        wts = zeros(0)
    end 
    return wts, pts
end

"""
    surf_wts, surf_pts = 
        cut_surf_quad(cell, phi, num_quad [, fit_degree=num_quad-1])

Returns a surface quadrature for the surface defined by `phi(x)=0` over the 
HyperRectangle `cell`, where `phi` is a level-set function.  If the surface is, 
for example, a plane parallel to one of the sides of `cell`, then we get a 
tensor-product Gauss-Legendre quadrature with `num_quad^(Dim-1)` points total, 
where `Dim` is the dimension of the domain.  The `fit_degree` parameter 
indicates the degree of the Bernstein polynomial that the algoim library uses 
to represent the level-set function `phi`. 

# Example
```julia-repl
julia> using CutQuad, RegionTrees

julia> using StaticArrays : SVector

julia> phi = x-> 4*(x[1] + 1)^2 + 36*(x[2] - 0.5)^2 - 9;

julia> cell = HyperRectangle(SVector{2}([0.0; 0.0]), SVector{2}([1.0; 1.0]));

julia> surf_wts, surf_pts = cut_surf_quad(cell, phi, 3, fit_degree=2)
```
"""
function cut_surf_quad(cell::HyperRectangle{1,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    surf_pts = Vector{Float64}(); surf_wts = Vector{Float64}()
    xwork = zeros(1) # allocating here has insignificant impact, it seems
    mod_levset[] = phi
    cphi = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))
    cut_surf_quad1d(cphi, num_quad, fit_degree,
                    Vector{Float64}(cell.origin),
                    Vector{Float64}(cell.origin + cell.widths),
                    xwork, surf_wts, surf_pts)
    return reshape(surf_wts, 1, :), reshape(surf_pts, 1, :)
end

function cut_surf_quad(cell::HyperRectangle{2,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    surf_pts = Vector{Float64}(); surf_wts = Vector{Float64}()
    xwork = zeros(2) # allocating here has insignificant impact, it seems
    mod_levset[] = phi
    cphi = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))
    cut_surf_quad2d(cphi, num_quad, fit_degree,
                    Vector{Float64}(cell.origin),
                    Vector{Float64}(cell.origin + cell.widths),
                    xwork, surf_wts, surf_pts)
    return reshape(surf_wts, 2, :), reshape(surf_pts, 2, :)
end

function cut_surf_quad(cell::HyperRectangle{3,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    surf_pts = Vector{Float64}(); surf_wts = Vector{Float64}()
    xwork = zeros(3) # allocating here has insignificant impact, it seems
    mod_levset[] = phi
    cphi = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))
    cut_surf_quad2d(cphi, num_quad, fit_degree,
                    Vector{Float64}(cell.origin),
                    Vector{Float64}(cell.origin + cell.widths),
                    xwork, surf_wts, surf_pts)
    return reshape(surf_wts, 3, :), reshape(surf_pts, 3, :)
end

export cut_cell_quad, cut_face_quad, cut_surf_quad

end
