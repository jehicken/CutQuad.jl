using CutQuad
using Documenter

DocMeta.setdocmeta!(CutQuad, :DocTestSetup, :(using CutQuad); recursive=true)

makedocs(;
    modules=[CutQuad],
    authors="Jason Hicken <jason.hicken@gmail.com> and contributors",
    repo="https://github.com/jehicken/CutQuad.jl/blob/{commit}{path}#{line}",
    sitename="CutQuad.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jehicken.github.io/CutQuad.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jehicken/CutQuad.jl",
    devbranch="main",
)
