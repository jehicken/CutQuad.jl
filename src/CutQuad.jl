module CutQuad

using RegionTrees, CxxWrap, AlgoimDiff_jll
if !isdefined(AlgoimDiff_jll, :libcutquad_path)
    error("Sorry!  AlgoimDiff_jll is not available on this platform")
end
@wrapmodule(libcutquad)
__init__() = @initcxx


"""
    wts, pts, surf_wts, surf_pts = 
        calc_cut_quad(rect, phi, num_quad [, fit_degree=num_quad-1])

Returns a volume and surface quadrature for the domain defined by the 
HyperRectangle `rect` and level-set function `phi`.  The canonical quadrature 
on a non-cut domain (i.e. `phi(x)< 0` for all `x`) uses a tensor-product 
Gauss-Legendre quadrature with `num_quad^dim` points total, where `dim` is the 
dimension of the domain.

**Note:** Users must supply the level-set function `phi` as a safe cfunction by 
wrapping their Julia function as shown in the example below.

# Example
```julia-repl
julia> using CutQuad, CxxWrap, RegionTrees

julia> using StaticArrays : SVector

julia> phi_julia = x-> 4*(x[1] + 1)^2 + 36*(x[2] - 0.5)^2 - 9;

julia> cphi = @safe_cfunction(phi_julia, Cdouble, (Vector{Float64},));

julia> rect = HyperRectangle(SVector{2}([0.0; 0.0]), SVector{2}([1.0; 1.0]));

julia> wts, pts, surf_wts, surf_pts = calc_cut_quad(rect, cphi, 3, fit_degree=5)
```
"""
function calc_cut_quad(rect::HyperRectangle{2,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    pts = Vector{Float64}(); wts = Vector{Float64}()
    surf_pts = Vector{Float64}(); surf_wts = Vector{Float64}()
    xwork = zeros(2) # allocating here has insignificant impact, it seems
    calc_cut_quad2d(phi, num_quad, fit_degree,
                    Vector{Float64}(rect.origin),
                    Vector{Float64}(rect.origin + rect.widths),
                    xwork, wts, pts, surf_wts, surf_pts)
    return wts, reshape(pts, 2, :), reshape(surf_wts, 2, :),
        reshape(surf_pts, 2, :)
end

function calc_cut_quad(rect::HyperRectangle{3,Float64}, phi,
                       num_quad::Int; fit_degree::Int=num_quad-1)
    @assert( num_quad > 0, "number of quadrature points must be positive." )
    @assert( fit_degree >= 0, "degree of polynomial fit must be non-negative." )
    pts = Vector{Float64}(); wts = Vector{Float64}()
    surf_pts = Vector{Float64}(); surf_wts = Vector{Float64}()
    xwork = zeros(3) # allocating here has insignificant impact, it seems
    calc_cut_quad3d(phi, num_quad, fit_degree,
                    Vector{Float64}(rect.origin),
                    Vector{Float64}(rect.origin + rect.widths),
                    xwork, wts, pts, surf_wts, surf_pts)
    return wts, reshape(pts, 3, :), reshape(surf_wts, 3, :),
        reshape(surf_pts, 3, :)
end

export calc_cut_quad

end
