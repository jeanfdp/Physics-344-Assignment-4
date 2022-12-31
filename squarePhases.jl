using Combinatorics, Graphs
using StatsBase, Random, Distributions
using DelimitedFiles, GraphIO
using Memoization

println(Threads.nthreads())

function loadEdges(file)
    return readdlm(file,',',Int64,'\n')
end

@memoize function splitEdges(n)
    return [loadEdges("./graphs/square/"*string(n)*"x"*string(n)*"/1.el")#Load strain 1 edges
    ,loadEdges("./graphs/square/"*string(n)*"x"*string(n)*"/2.el")]#Load strain 2 edges
end

@memoize function underlyingGraph(n)#Calculate the 'total' graph, where all possible edges are present
    edges=splitEdges(n)
    g = SimpleGraph(UInt32(n^2))
    for i in 1:div(length(edges[1]),2)
        add_edge!(g,edges[1][i,1],edges[1][i,2])
    end
    for i in 1:div(length(edges[2]),2)
        add_edge!(g,edges[2][i,1],edges[2][i,2])
    end
    return g
end

function op1(g)
    return float(maximum(length.(connected_components(g))))#return amount of vertices in largest component as a float
end

function op2(g)
    verts=findall(x->x==8,degree(g))#Find all vertices of degree 8
    if length(verts)>0
        return op1(induced_subgraph(g,verts)[1])#Find largest component size in subgraph of vertices with degree 8
    else
        return 0
    end
end

function op3(g)
    verts=findall(x->x==4,degree(g))#Find all vertices of degree 4
    if length(verts)>0
        return op1(induced_subgraph(g,verts)[1])#Find largest component size in subgraph of vertices with degree 4
    else
        return 0
    end
end

function op4(g,ug)
    verts=findall(x->x==0,degree(g))#Find all vertices of degree 0
    if length(verts)>0
        return op1(induced_subgraph(ug,verts)[1])#Find largest component size in subgraph of vertices with degree 0 of the underluing graph
    else
        return 0
    end
end

function findOrderParams(n,β,μ,k,edges,ug,tmpgs)
    out=[Threads.Atomic{Float64}(0) for _=1:4]#Will store order parameter estimations, and is safe for multithreading
    (dist1,dist2)=(Bernoulli(1-1/(1+exp(β*(μ)))),Bernoulli(1-1/(1+exp(β*(μ-1)))))#The Bernoulli distributions that will be used to see if a given edge should be present in a typical graph
    Threads.@threads for _=1:k
        tmpgs[Threads.threadid()]=SimpleGraph(UInt32(n^2))#Initialise empty graph with the correct amount of vertices
        for i in 1:div(length(edges[1]),2)
            if rand(dist1)
                add_edge!(tmpgs[Threads.threadid()],edges[1][i,1],edges[1][i,2])#Add strain 1 edge if applicable
            end
        end
        for i in 1:div(length(edges[2]),2)
            if rand(dist2)
                add_edge!(tmpgs[Threads.threadid()],edges[2][i,1],edges[2][i,2])#Add strain 2 edge if applicable
            end
        end
        Threads.atomic_add!(out[1],op1(tmpgs[Threads.threadid()])/(k*n^2))#Update averages
        Threads.atomic_add!(out[2],op2(tmpgs[Threads.threadid()])/(k*n^2))
        Threads.atomic_add!(out[3],op3(tmpgs[Threads.threadid()])/(k*n^2))
        Threads.atomic_add!(out[4],op4(tmpgs[Threads.threadid()],ug)/(k*n^2))
    end
    return [out[1][],out[2][],out[3][],out[4][]]
end

const pointfile=string(ARGS[1])
const n=parse(Int,ARGS[2])
const k=parse(Int,ARGS[3])

const edges=splitEdges(n)
const ug=underlyingGraph(n)

const points=readdlm(pointfile,',',Float64,'\n')

out =zeros((div(length(points),2),4))

tmpgs=[SimpleGraph(UInt32(n^2)) for _=1:Threads.nthreads()]

t=@time for i=1:div(length(points),2)
    tmp=findOrderParams(n,points[i,1],points[i,2],k,edges,ug,tmpgs)
    for j=1:4
        out[i,j]=tmp[j]
    end
end

open("out_"*string(n)*"_"*string(k)*"_"*pointfile,"w") do io
    writedlm(io,out,',')
end

println(t)