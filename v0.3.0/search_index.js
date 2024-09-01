var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = CutQuad","category":"page"},{"location":"#CutQuad","page":"Home","title":"CutQuad","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for CutQuad.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [CutQuad]","category":"page"},{"location":"#CutQuad.cut_cell_quad-Tuple{RegionTrees.HyperRectangle{1, Float64}, Any, Int64}","page":"Home","title":"CutQuad.cut_cell_quad","text":"wts, pts = cut_cell_quad(cell, phi, num_quad [, fit_degree=num_quad-1])\n\nReturns a volume quadrature for the domain defined by the HyperRectangle cell  and level-set function phi.  The canonical quadrature on a non-cut domain (i.e. phi(x) > 0 for all x) uses a tensor-product Gauss-Legendre quadrature  with num_quad^Dim points total, where Dim is the dimension of the domain. The fit_degree parameter indicates the degree of the Bernstein polynomial that the algoim library uses to represent the level-set function phi.  \n\nExample\n\njulia> using CutQuad, RegionTrees\n\njulia> using StaticArrays: SVector\n\njulia> phi = x-> 4*(x[1] + 1)^2 + 36*(x[2] - 0.5)^2 - 9;\n\njulia> cell = HyperRectangle(SVector{2}([0.0; 0.0]), SVector{2}([1.0; 1.0]));\n\njulia> wts, pts = cut_cell_quad(cell, phi, 3, fit_degree=2)\n\n\n\n\n\n","category":"method"},{"location":"#CutQuad.cut_face_quad-Union{Tuple{Dim}, Tuple{RegionTrees.HyperRectangle{Dim, Float64}, Int64, Any, Int64}} where Dim","page":"Home","title":"CutQuad.cut_face_quad","text":"wts, pts = cut_face_quad(face, dir, phi, num_quad [, fit_degree=num_quad-1])\n\nReturns a quadrature for the face defined by the HyperRectangle face and  level-set function phi.  The face has a unit normal whose only non-zero component is in the direction dir; furthermore, if the spatial dimension is Dim –- which is determined by the parameter in face –- the HyperRectangle should have a width of zero in the direction dir.  The canonical quadrature on a non-cut face (i.e. phi(x) > 0 for all x over face) uses a tensor-product Gauss-Legendre quadrature with num_quad^(Dim-1) points total. The fit_degree parameter indicates the degree of the Bernstein polynomial that the algoim library uses to represent the level-set function phi.  \n\nExample\n\njulia> using CutQuad, RegionTrees\n\njulia> using StaticArrays: SVector\n\njulia> phi = x-> 4*(x[1] + 1)^2 + 36*(x[2] - 0.5)^2 - 9;\n\njulia> face = HyperRectangle(SVector{2}([0.0; 0.0]), SVector{2}([0.0; 1.0]));\n\njulia> face_wts, face_pts = cut_face_quad(face, 1, phi, 3, fit_degree=2)\n\n\n\n\n\n","category":"method"},{"location":"#CutQuad.cut_surf_quad-Tuple{RegionTrees.HyperRectangle{1, Float64}, Any, Int64}","page":"Home","title":"CutQuad.cut_surf_quad","text":"surf_wts, surf_pts = \n    cut_surf_quad(cell, phi, num_quad [, fit_degree=num_quad-1])\n\nReturns a surface quadrature for the surface defined by phi(x)=0 over the  HyperRectangle cell, where phi is a level-set function.  If the surface is,  for example, a plane parallel to one of the sides of cell, then we get a  tensor-product Gauss-Legendre quadrature with num_quad^(Dim-1) points total,  where Dim is the dimension of the domain.  The fit_degree parameter  indicates the degree of the Bernstein polynomial that the algoim library uses  to represent the level-set function phi. \n\nExample\n\njulia> using CutQuad, RegionTrees\n\njulia> using StaticArrays: SVector\n\njulia> phi = x-> 4*(x[1] + 1)^2 + 36*(x[2] - 0.5)^2 - 9;\n\njulia> cell = HyperRectangle(SVector{2}([0.0; 0.0]), SVector{2}([1.0; 1.0]));\n\njulia> surf_wts, surf_pts = cut_surf_quad(cell, phi, 3, fit_degree=2)\n\n\n\n\n\n","category":"method"}]
}
