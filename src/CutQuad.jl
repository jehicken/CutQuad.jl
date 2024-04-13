module CutQuad

using RegionTrees, CxxWrap, AlgoimDiff_jll
if !isdefined(AlgoimDiff_jll, :libcutquad_path)
    error("Sorry!  AlgoimDiff_jll is not available on this platform")
end
@wrapmodule(libcutquad)
# The following @wrapmodule is for CxxWrap@0.14.0 and above 
# @wrapmodule(AlgoimDiff_jll.get_libcutquad_path)
__init__() = @initcxx

# using RegionTrees, CxxWrap
# @wrapmodule(joinpath("/home/jehicken/Libraries/algoim","libcutquad"))
# __init__() = @initcxx 

# Module variable used to store a reference to the levset;
# this is needed for the @safe_cfunction macro
const mod_levset = Ref{Any}()

"""
    wts, pts = cut_cell_quad(rect, phi, num_quad [, fit_degree=num_quad-1])

Returns a volume quadrature for the domain defined by the HyperRectangle `rect` 
and level-set function `phi`.  The canonical quadrature on a non-cut domain
(i.e. `phi(x) > 0` for all `x`) uses a tensor-product Gauss-Legendre quadrature 
with `num_quad^dim` points total, where `dim` is the dimension of the domain.

# Example
```julia-repl
julia> using CutQuad, RegionTrees

julia> using StaticArrays : SVector

julia> phi = x-> 4*(x[1] + 1)^2 + 36*(x[2] - 0.5)^2 - 9;

julia> rect = HyperRectangle(SVector{2}([0.0; 0.0]), SVector{2}([1.0; 1.0]));

julia> wts, pts = cut_cell_quad(rect, phi, 3, fit_degree=2)
```
"""
function cut_cell_quad(rect::HyperRectangle{1,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    pts = Vector{Float64}(); wts = Vector{Float64}()
    xwork = zeros(1) # allocating here has insignificant impact, it seems
    mod_levset[] = phi
    cphi = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))
    cut_cell_quad1d(cphi, num_quad, fit_degree,
                    Vector{Float64}(rect.origin),
                    Vector{Float64}(rect.origin + rect.widths),
                    xwork, wts, pts)
    return wts, reshape(pts, 1, :)
end

function cut_cell_quad(rect::HyperRectangle{2,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    pts = Vector{Float64}(); wts = Vector{Float64}()
    xwork = zeros(2) # allocating here has insignificant impact, it seems
    mod_levset[] = phi
    cphi = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))
    cut_cell_quad2d(cphi, num_quad, fit_degree,
                    Vector{Float64}(rect.origin),
                    Vector{Float64}(rect.origin + rect.widths),
                    xwork, wts, pts)
    return wts, reshape(pts, 2, :)
end

function cut_cell_quad(rect::HyperRectangle{3,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    pts = Vector{Float64}(); wts = Vector{Float64}()    
    xwork = zeros(3) # allocating here has insignificant impact, it seems
    mod_levset[] = phi
    cphi = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))
    cut_cell_quad3d(cphi, num_quad, fit_degree,
                    Vector{Float64}(rect.origin),
                    Vector{Float64}(rect.origin + rect.widths),
                    xwork, wts, pts)
    return wts, reshape(pts, 3, :)
end

"""
    surf_wts, surf_pts = 
        cut_surf_quad(rect, phi, num_quad [, fit_degree=num_quad-1])

Returns a surface quadrature for the surface defined by `phi(x)=0` over the 
HyperRectangle `rect`, where `phi` is a level-set function.  If the surface is, 
for example, a plane parallel to one of the sides of `rect`, then we get a 
tensor-product Gauss-Legendre quadrature with `num_quad^(dim-1)` points total, 
where `dim` is the dimension of the domain.

# Example
```julia-repl
julia> using CutQuad, RegionTrees

julia> using StaticArrays : SVector

julia> phi = x-> 4*(x[1] + 1)^2 + 36*(x[2] - 0.5)^2 - 9;

julia> rect = HyperRectangle(SVector{2}([0.0; 0.0]), SVector{2}([1.0; 1.0]));

julia> surf_wts, surf_pts = cut_surf_quad(rect, phi, 3, fit_degree=2)
```
"""
function cut_surf_quad(rect::HyperRectangle{1,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    surf_pts = Vector{Float64}(); surf_wts = Vector{Float64}()
    xwork = zeros(1) # allocating here has insignificant impact, it seems
    mod_levset[] = phi
    cphi = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))
    cut_surf_quad1d(cphi, num_quad, fit_degree,
                    Vector{Float64}(rect.origin),
                    Vector{Float64}(rect.origin + rect.widths),
                    xwork, surf_wts, surf_pts)
    return reshape(surf_wts, 1, :), reshape(surf_pts, 1, :)
end

function cut_surf_quad(rect::HyperRectangle{2,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    surf_pts = Vector{Float64}(); surf_wts = Vector{Float64}()
    xwork = zeros(2) # allocating here has insignificant impact, it seems
    mod_levset[] = phi
    cphi = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))
    cut_surf_quad2d(cphi, num_quad, fit_degree,
                    Vector{Float64}(rect.origin),
                    Vector{Float64}(rect.origin + rect.widths),
                    xwork, surf_wts, surf_pts)
    return reshape(surf_wts, 2, :), reshape(surf_pts, 2, :)
end

function cut_surf_quad(rect::HyperRectangle{3,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    surf_pts = Vector{Float64}(); surf_wts = Vector{Float64}()
    xwork = zeros(3) # allocating here has insignificant impact, it seems
    mod_levset[] = phi
    cphi = @safe_cfunction( x -> mod_levset[](x), Float64, (Vector{Float64},))
    cut_surf_quad2d(cphi, num_quad, fit_degree,
                    Vector{Float64}(rect.origin),
                    Vector{Float64}(rect.origin + rect.widths),
                    xwork, surf_wts, surf_pts)
    return reshape(surf_wts, 3, :), reshape(surf_pts, 3, :)
end

export cut_cell_quad, cut_surf_quad

end
