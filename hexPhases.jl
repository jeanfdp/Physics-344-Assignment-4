using Combinatorics, Graphs
using StatsBase, Random, Distributions
using DelimitedFiles, GraphIO
using Memoization

println(Threads.nthreads())

function loadEdges(file)
    return readdlm(file,',',Int64,'\n')
end

@memoize function splitEdges(n)
    return [loadEdges("./graphs/hex/"*string(n)*"x"*string(n)*"/1.el")#Load strain 1 edges
    ,loadEdges("./graphs/hex/"*string(n)*"x"*string(n)*"/2.el"),#Load strain 2 edges
    loadEdges("./graphs/hex/"*string(n)*"x"*string(n)*"/3.el")]#Load strain 3 edges
end

@memoize function underlyingGraph(n)#Calculate the 'total' graph, where all possible edges are present
    g=SimpleGraph(UInt32(vs(n)))
    edges=splitEdges(n)
    for i in 1:div(length(edges[1]),2)
        add_edge!(g,edges[1][i,1],edges[1][i,2])
    end
    for i in 1:div(length(edges[2]),2)
        add_edge!(g,edges[2][i,1],edges[2][i,2])
    end
    for i in 1:div(length(edges[3]),2)
        add_edge!(g,edges[3][i,1],edges[3][i,2])
    end
    return g
end

@memoize function vs(n)#Calculate the amount of vertices for a given n value
    edges=splitEdges(n)
    return maximum([maximum(edges[1]),maximum(edges[2]),maximum(edges[3])])
end

function op1(g)
    return float(maximum(map(x->length(x),connected_components(g))))#return amount of vertices in largest component
end

function op2(g)
    out =[]
    for n=1:12
        verts=findall(x->x==n,degree(g))#Find all vertices of degree n
        if length(verts)>0
            push!(out,op1(induced_subgraph(g,verts)[1]))
        else
            push!(out,0.)
        end
    end
    return out
end

function op3(g,ug)
    verts=findall(x->x==0,degree(g))#Find all vertices of degree 0
    if length(verts)>0
        return op1(induced_subgraph(ug,verts)[1])#Find largest component size in subgraph of vertices with degree 0
    else
        return 0
    end
end

function findOrderParams(n,β,μ,k,edges,ug,tmpgs)
    out=[Threads.Atomic{Float64}(0) for _=1:14]
    (dist1,dist2,dist3)=(Bernoulli(1-1/(1+exp(β*(μ)))),Bernoulli(1-1/(1+exp(β*(μ-1)))),Bernoulli(1-1/(1+exp(β*(μ-2)))))
    Threads.@threads for _=1:k
        tmpgs[Threads.threadid()]=SimpleGraph(UInt32(vs(n)))
        for i in 1:div(length(edges[1]),2)
            if rand(dist1)
                add_edge!(tmpgs[Threads.threadid()],edges[1][i,1],edges[1][i,2])
            end
        end
        for i in 1:div(length(edges[2]),2)
            if rand(dist2)
                add_edge!(tmpgs[Threads.threadid()],edges[2][i,1],edges[2][i,2])
            end
        end
        for i in 1:div(length(edges[3]),2)
            if rand(dist3)
                add_edge!(tmpgs[Threads.threadid()],edges[3][i,1],edges[3][i,2])
            end
        end
        Threads.atomic_add!(out[1],op1(tmpgs[Threads.threadid()])/(k*vs(n)))
        ops=op2(tmpgs[Threads.threadid()])
        for i=1:12
            Threads.atomic_add!(out[i+1],ops[i]/(k*vs(n)))
        end
        Threads.atomic_add!(out[14],op3(tmpgs[Threads.threadid()],ug)/(k*vs(n)))
    end
    return [out[i][] for i=1:14]
end

const pointfile=string(ARGS[1])
const n=parse(Int,ARGS[2])
const k=parse(Int,ARGS[3])

const edges=splitEdges(n)
const ug=underlyingGraph(n)

const points=readdlm(pointfile,',',Float64,'\n')

out =zeros((div(length(points),2),14))

tmpgs=[SimpleGraph(UInt32(vs(n))) for _=1:Threads.nthreads()]

t=@time for i=1:div(length(points),2)
    tmp=findOrderParams(n,points[i,1],points[i,2],k,edges,ug,tmpgs)
    for j=1:14
        out[i,j]=tmp[j]
    end
end

open("out_"*string(n)*"_"*string(k)*"_"*pointfile,"w") do io
    writedlm(io,out,',')
end

println(t)