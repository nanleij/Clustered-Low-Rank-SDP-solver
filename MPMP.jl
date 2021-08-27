module MPMP

#NOTE: I used JuliaFormatter to format (most of) the file, but I'm not sure whether I'm happy with the results

using AbstractAlgebra, Combinatorics #AbstractAlgebra is not really needed right?
using GenericLinearAlgebra, LinearAlgebra #need GenericLinearAlgebra for BigFloat support (eigmin)

using BlockDiagonals
using Printf
using GenericSVD #SVD of bigfloat matrix, for prepareabc with rank>1 structure
using Arblib #lightweight implementation of Arb, in development. We are using it for matrix multiplication, and possibly for cholesky if that works

#for extending to Arblib and BlockDiagonal
import LinearAlgebra.dot, LinearAlgebra.cholesky
import Base.abs, Base.max

const T = BigFloat

export solvempmp, solverank1sdp, get_block_info, prepareabc, laguerrebasis


## Functions for input to the solver (i.e making bases or sample points)
"""Make the monomial basis of the polynomial ring with maximum total_degree d"""
function make_monomial_basis(poly_ring, d)
    #NOTE: this is the monomial basis, so in general a very bad choice
    #works for any number of variables
    n = nvars(poly_ring)
    vars = gens(poly_ring)
    q = zeros(poly_ring, binomial(n + d, d))#n+d choose d basis polynomials
    q_idx = 1
    for k = 0:d
        for exponent in multiexponents(n, k) #exponent vectors of size n with total degree k.
            temp_pol = MPolyBuildCtx(poly_ring)
            push_term!(temp_pol, poly_ring(1)(zeros(n)...), exponent)
            temp_pol = finish(temp_pol)
            q[q_idx] = temp_pol
            q_idx += 1
        end
    end
    return q
end

function laguerrebasis(k::Integer, alpha, x)
    #Davids function
    v = Vector{typeof(one(alpha) * one(x))}(undef, 1 + k)
    v[1] = one(x)
    k == 0 && return v
    v[2] = 1 + alpha - x
    k == 1 && return v
    for l = 2:k
        v[l+1] = 1 // big(l) * ((2l - 1 + alpha - x) * v[l] - (l + alpha - 1) * v[l-1])
    end
    return v
end

function jacobi_basis(d::Integer, alpha, beta, x, normalized = true)
    q = Vector{typeof(one(alpha) * one(x))}(undef, d + 1)
    q[1] = one(x)
    d == 0 && return q
    q[2] = x # normalized
    if !normalized
        q[2] *= (alpha + 1)
    end
    d == 1 && return q
    for k = 2:d
        #what if alpha+beta = -n for some integer n>=1
        q[k+1] =
            (2 * k + alpha + beta - 1) /
            BigFloat(2k * (k + alpha + beta) * (2k + alpha + beta - 2)) *
            ((2 * k + alpha + beta) * (2k + alpha + beta - 2) * x + beta^2 - alpha^2) *
            q[k] + -2 * (k + alpha - 1) * (k + beta - 1) * (2 * k + alpha + beta) * q[k-1]
        # q[k+1] = (2*k+alpha+ beta-1)*(k+alpha)/BigFloat(k*(k+2*alpha))*x*q[k] - (k+alpha-1)*(k+alpha)/BigFloat(k*(k+2*alpha))*q[k-1]
    end
    return q
end

"Basis for the Gegenbauer polynomials in dimension n up to degree k.
 This is the Gegenbauer polynomial with parameter lambda = n/2-1,
 or the Jacobi polynomial with parameters alpha = beta = (n-3)/2.
 Normalized to evaluate to 1 at 1.
 Taken from arxiv/2001.00256, ancillary files, SemidefiniteProgramming.jl"
function gegenbauer_basis(k, n, x)
    v = Vector{typeof(one(x))}(undef, 1 + k)
    v[1] = one(x)
    k == 0 && return v
    v[2] = x
    k == 1 && return v
    for l = 2:k
        v[l+1] = (2l + n - 4) // (l + n - 3) * x * v[l] - (l - 1) // (l + n - 3) * v[l-1]
    end
    v
end

function create_sample_points(n, d)
    #rational points in the unit simplex with denominator d
    #probably not very efficient, but I dont know how to do it better for general n.
    x = [zeros(BigFloat, n) for i = 1:binomial(n + d, d)] #need n+d choose d points for a unisolvent set, if symmetry is not used.
    idx = 1
    for I in CartesianIndices(ntuple(k -> 0:d, n)) #all tuples with elements in 0:d of length n
        if sum(Tuple(I)) <= d #in unit simplex
            x[idx] = [i / BigFloat(d) for i in Tuple(I)]
            idx += 1
        end
    end
    return x
end

function create_sample_points_2d(d)
    #padua points:
    z = [Array{BigFloat}(undef, 2) for i = 1:binomial(2 + d, d)]
    z_idx = 1
    for j = 0:d
        delta_j = j % 2 == d % 2 == 1 ? 1 : 0
        mu_j = cospi(j / d)
        for k = 1:(div(d, 2)+1+delta_j)
            eta_k = j % 2 == 1 ? cospi((2 * k - 2) / (d + 1)) : cospi((2 * k - 1) / (d + 1))
            z[z_idx] = [mu_j, eta_k]
            z_idx += 1
        end
    end
    return z
end

function create_sample_points_3d(d; pairs = [(1, 3), (3, 2), (2, 1)])#of pairs tested was this the best one. Good for odd n. This approach does not really work for even n.
    d % 2 == 0 && println(
        "n should be odd for the sample points to be good. Consider using different sample points.",
    )
    # extension of padua & chebyshev points. Similar to how padua points are an extension of chebyshev points
    pad = create_sample_points_2d(d) #size (n+1)*(n+2)/2
    pad_div = [pad[1:3:end], pad[2:3:end], pad[3:3:end]] # should we divide it up in a different way?
    ch = create_sample_points_chebyshev(d + 2) #size (n+3)
    cheb_div = [ch[1:3:end], ch[2:3:end], ch[3:3:end]]
    total_points =
        [similar([pad[1]..., ch[1]]) for i = 1:div((d + 1) * (d + 2) * (d + 3), 6)]
    cur_point = 1
    for pair in pairs
        for p1 in pad_div[pair[1]]
            for p2 in cheb_div[pair[2]]
                total_points[cur_point] = [p1..., p2]
                cur_point += 1
            end
        end
    end
    return total_points
end

function points_X_general(n, d)# sometimes good, not always.
    #works with n=4: d=2,3,5,11. Not for d=4,6,7,8
    if n == 2
        return MPMP.create_sample_points_2d(d)
    end
    Xn_1 = points_X_general(n - 1, d)
    cheb = MPMP.create_sample_points_chebyshev(d + n - 1)
    println(length(Xn_1))
    println(length(cheb))
    X_div = [Xn_1[i:n:end] for i = 1:n]
    cheb_div = [cheb[i:n:end] for i = 1:n]
    total_points = [similar([Xn_1[1]..., cheb[1]]) for i = 1:binomial(n + d, d)]
    cur_point = 1
    for i = 1:n
        j = i == 1 ? n : i - 1
        for p1 in X_div[i]
            for p2 in cheb_div[j]
                total_points[cur_point] = [p1..., p2]
                cur_point += 1
            end
        end
    end
    return total_points
end


function create_sample_points_1d(d)
    #as done in simmons duffin: ('rescaled Laguerre')
    # x[k] = sqrt(pi)/(64*(d+1)*log( 3- 2*sqrt(2))) * (-1+4*k)^2, with k=0:d
    constant = -sqrt(BigFloat(pi)) / (64 * (d + 1) * log(3 - 2 * sqrt(BigFloat(2))))
    x = zeros(BigFloat, d + 1)
    for k = 0:d
        x[k+1] = constant * (-1 + 4 * k)^2
    end
    return x
end

function create_sample_points_chebyshev(d, a = -1, b = 1)
    #roots of chebyshev polynomials of the first kind, unisolvent for polynomials up to degree d
    return [
        (a + b) / BigFloat(2) +
        (b - a) / BigFloat(2) * cos((2k - 1) / BigFloat(2(d + 1)) * BigFloat(pi)) for
        k = 1:d+1
    ]
end

function create_sample_points_chebyshev_mod(d, a = -1, b = 1)
    #roots of chebyshev polynomials of the first kind, divided by cos(pi/2(d+1)) to get a lower lebesgue constant
    return [
        (a + b) / BigFloat(2) +
        (b - a) / BigFloat(2) * cos((2k - 1) / BigFloat(2(d + 1)) * BigFloat(pi)) /
        cos(BigFloat(pi) / 2(d + 1)) for k = 1:d+1
    ]
end

## Functions for the solver. This part is commented more thoroughly

#extending LinearAlgebra.dot for the BlockDiagonal matrices.
function LinearAlgebra.dot(A::BlockDiagonal, B::BlockDiagonal)
    #assume that A and B have the same blockstructure
    @assert  length(blocks(A)) == length(blocks(B))
    sum(dot(a, b) for (a, b) in zip(blocks(A), blocks(B)))
end
#Extending the LinearAlgebra.dot for ArbMatrices. Arblib does not have an inner product for matrices
function LinearAlgebra.dot(A::ArbMatrix, B::ArbMatrix)
    @assert size(A) == size(B)
    res = Arb(0, prec = precision(A))
    for i = 1:size(A, 1)
        for j = 1:size(A, 2)
            Arblib.addmul!(res, ref(A, i, j), ref(B, i, j))
        end
    end
    return res
end

#NOTE: currently the format is g qq^T ⊗ Pi . The eigenvalues of Pi and the sign of g are stored together.
#       Now that we have the eigenvalues of Pi also in A_sign, we can just as well put the whole G there.
# other possibility is to make the structure qq^T ⊗ Pi, with g included in Pi
function prepareabc(
    M, # Vector of matrix polynomials
    G, # Vector of polynomials
    q, # Vector of polynomials
    x, # Vector of points
    δ = -1,# maximum degree. negative numbers-> use 2* the total degree of q[end]
    Pi = nothing;         #polynomial matrices, as much as G has. We will use A_(jrsk) = sum_l Tr(Q ⊗ E_rs) where Q = ∑_η λ_η(x_k)(√G[l](x_k)q(x_k) ⊗ v_η(x_k)) (√G[l](x_k)q(x_k) ⊗ v_η(x_k))^T
    prec = precision(BigFloat),
    all_of_Pi = true, #When true, we use also parts of Pi when they have lower degree; not only whole Pi blocks
    threshold = BigFloat(10)^(-70),
    qp_precomp = nothing
)
    #So λ_η sign(G[l](x_k)) is stored in A_sign[l,k][η] = H_(l,k,η), and the vectors (√G[l](x_k)q(x_k) ⊗ v_η(x_k)) in A[l,k][η]
    # Assumptions:
    # 1) all elements of M are the same size (true for a constraint)
    # 2) M[i][r,s] all have the same number of variables (for all possible i,r,s)
    # 3) The basis is correct for the given symmetry (in Pi)
    # 4) (if no max degree given) q contains precisely enough polynomomials to span the whole space (last polynomial has degree δ/2)

    m = size(M[1], 1)
    n = nvars(parent(M[1][1, 1])) #the number of variables of the polynomial ring of M[1][1,1].
    if δ < 0
        δ = 2 * total_degree(q[end]) #assumption: q is exactly long enough
    end

    if isnothing(Pi)
        Pi_vecs = [[[T(1)]] for l in G, k = 1:length(x)]
        Pi_vals = [[T(1)] for l in G, k = 1:length(x)]
        deg_Pi = [0 for l in G]
        deg_Pi_vec = [[0] for l in G]
    else
        #We use an SVD to get eigenvalues and vectors. getting eigenvectors for bigfloatmatrices is not possible with eigen(M) or eigvecs(M)
        svd_decomps = [
            svd([Pi[l][i, j](x[k]...) for i = 1:size(Pi[l], 1), j = 1:size(Pi[l], 2)])
            for l = 1:length(G), k = 1:length(x)
        ]
        Pi_vecs = [
            [svd_decomps[l, k].U[:, r] for r = 1:size(Pi[l], 1)] for l = 1:length(G),
            k = 1:length(x)
        ]
        Pi_vals = [
            [
                sign(dot(svd_decomps[l, k].U[:, r], svd_decomps[l, k].Vt[:, r])) *
                svd_decomps[l, k].S[r] for r = 1:size(Pi[l], 1)
            ] for l = 1:length(G), k = 1:length(x)
        ]
        deg_Pi = [
            max(
                [
                    total_degree(Pi[l][i, j]) for i = 1:size(Pi[l], 1) for
                    j = 1:size(Pi[l], 2)
                ]...,
            ) for l = 1:length(G)
        ]
        deg_Pi_vec =
            [[total_degree(Pi[l][i, i]) for i = 1:size(Pi[l], 1)] for l = 1:length(G)]
    end

    # We store the last occurance of a degree in the basis.
    # This is needed for symmetries, where the number of required basis polynomials can be (much) smaller than (n+d choose d)
    degrees = ones(Int64, div(δ, 2) + 1) #maximum degree needed is δ/2. everything is an index, so at least 1
    cur_deg = 0
    all_degrees = [total_degree(q[i]) for i = 1:length(q)]
    #check for monotonicity:
    for i = 1:length(all_degrees)-1
        if all_degrees[i] > all_degrees[i+1]
            println(
                "Degrees are not monotone. The program will (most probably) not be correct if you don't fix this",
            )
        end
    end
    last_deg = [findlast(x -> x == i, all_degrees) for i = 0:div(δ, 2)] #at place d+1: the last last index i such that deg(q[i]) = d
    #last_deg[i] == nothing if degree i did not occur in all_degrees
    #We change the nothings into the previous one (For some symmetries, not every degree is part of the basis)
    for i = 1:length(last_deg)
        if isnothing(last_deg[i])
            last_deg[i] = last_deg[i-1] #always works if 1 is the first entry and the degrees are monotone
        end
    end

    # We can even put the whole G[l](x[k]...) in the 'sign'. We already have the eigenvalues of the Pi there
    # either way, better to be consistent (either also G in A_sign, or only the signs -> ev of Pi in A). I'm not sure whether it will matter during solving
    A_sign = [
        [
            Arb(Pi_vals[l, k][r] * sign(G[l](x[k]...)), prec = prec) for
            r = 1:length(Pi_vals[l, k])
        ] for l = 1:length(G), k = 1:length(x)
    ]
    if !all_of_Pi
        if !isnothing(qd_precomp)
            A = [
                [
                    ArbMatrix(
                        kron(
                            [
                                qp_precomp[k,d] * sqrt(abs(G[l](x[k]...))) for
                                d = 1:last_deg[div(δ - total_degree(G[l]) - deg_Pi[l], 2)+1]
                            ],
                            Pi_vecs[l, k][r],
                        ),
                        prec = prec,
                    ) for r = 1:length(Pi_vecs[l, k])
                ] for l = 1:length(G), k = 1:length(x)
            ]
        else
            A = [
                [
                    ArbMatrix(
                        kron(
                            [
                                q[d](x[k]...) * sqrt(abs(G[l](x[k]...))) for
                                d = 1:last_deg[div(δ - total_degree(G[l]) - deg_Pi[l], 2)+1]
                            ],
                            Pi_vecs[l, k][r],
                        ),
                        prec = prec,
                    ) for r = 1:length(Pi_vecs[l, k])
                ] for l = 1:length(G), k = 1:length(x)
            ]
        end
    else #all_of_Pi = true
        A = [
            Vector{ArbMatrix}(undef,length(Pi_vecs[l,k])) for l = 1:length(G), k = 1:length(x)
        ]
        for l = 1:length(G), k = 1:length(x)
            #make A[k,l]
            #We do the kronecker product manually, because some of the rows can use extra basis elements
            #Other way: do the full kronecker product, and select the rows manually.
            for r = 1:length(Pi_vecs[l, k])
                #kronecker product
                if !isnothing(qp_precomp)
                    vec_cur_r = [
                        Pi_vecs[l, k][r][Pi_deg_idx] * qp_precomp[k,d] * sqrt(abs(G[l](x[k]...))) for Pi_deg_idx = 1:length(deg_Pi_vec[l])
                        for d =
                            1:last_deg[div(
                                δ - total_degree(G[l]) - deg_Pi_vec[l][Pi_deg_idx],
                                2,
                            )+1]
                    ]
                else
                    vec_cur_r = [
                        Pi_vecs[l, k][r][Pi_deg_idx] * q[d](x[k]...) * sqrt(abs(G[l](x[k]...))) for Pi_deg_idx = 1:length(deg_Pi_vec[l])
                        for d =
                            1:last_deg[div(
                                δ - total_degree(G[l]) - deg_Pi_vec[l][Pi_deg_idx],
                                2,
                            )+1]
                    ]
                end
                A[l,k][r] = ArbMatrix(vec_cur_r, prec = prec)
            end
        end
    end
    for l = 1:length(G), k = 1:length(x)
        #Remove the eigenvalues/vectors of A which are almost 0
        keep_idx = [i for i = 1:length(A_sign[l, k]) if abs(A_sign[l, k][i]) > threshold]
        A_sign[l, k] = A_sign[l, k][keep_idx]
        A[l, k] = A[l, k][keep_idx]
    end

    # A[l,k][r] is the vector v_{j,l,k,r} for this constraint j
    # A_sign[l,k][r] gives the sign and the r'th eigenvalue of Pi. i.e Q = \sum_r A_sign[l,k][r] A[l,k][r] *A[l,k][r]'
    B = ArbMatrix(
        vcat(
            [
                transpose([T(-M[i][r, s](x[k]...)) for i = 2:length(M)]) for r = 1:m for
                s = 1:r for k = 1:length(x)
            ]...,
        ),
        prec = prec,
    )

    c = ArbMatrix(
        vcat([[T(M[1][r, s](x[k]...))] for r = 1:m for s = 1:r for k = 1:length(x)]...),
        prec = prec,
    )
    #indexing:
    # A[l,k][rnk]::ArbMatrix(., 1)
    # B::ArbMatrix
    # c::ArbMatrix(., 1)
    # A_sign[l,k][rnk]
    return A, B, c, A_sign
end

# function distribute_weights(weights,n)
#     #Distribute the weights over the n clusters
#     #we need n subsets of [1,..., length(weights)] such that
#     #   1) the max difference in cardinality is 1 between subsets
#     #   2) the maximum weight is (approximately) minimal
#     #   3) the subsets are disjoint and their union is 1,..., len(weights)
#     sets = [Int[] for i=1:n]# one set for each core
#     set_weights = [eltype(weights)[] for i=1:n]
#     max_lengths = div(length(weights),n)+1
#     n_max_lengths = max_lengths*n-length(weights)
#     sorted_weights = sort([(weights[i],i) for i=1:length(weights)],rev=true)#sort on the weight but keep track of the index
#     for i=1:length(weights)
#         cur_i = sorted_weights[i][2]
#     end
# end

function distribute_weights_swapping(weights,n;nswaps = length(weights)^2)
    #seems to work quite well for such a simple algorihtm
    #we first 'normally' distribute the weights, then try to improve it a number of times by swapping stuff between large and small sets
    step = div(length(weights),n)+1 #first number of steps have this size
    nstep = n-(step*n-length(weights))
    sets = vcat([collect((i-1)*step+1:i*step) for i=1:nstep],
            [collect(nstep*step+(i-1)*(step-1)+1:nstep*step+i*(step-1)) for i=1:n-nstep])
    set_weights = [sum(weights[sets[i]]) for i=1:length(sets)]
    index_set = 1
    index_el = 1
    for k=1:nswaps #try some swaps. In principle, moving an element from a set with step elements to a set with step-1 elements is also allowed
        max_set = sort([(set_weights[i],i) for i=1:length(set_weights)],rev=true)[index_set][2]
        max_el = sets[max_set][sort([(weights[sets[max_set]][i],i) for i=1:length(weights[sets[max_set]])],rev=true)[index_el][2]]
        min_set = argmin(set_weights)
        min_el = sets[min_set][argmin(weights[sets[min_set]])]
        #see if the swap works: the min_set should not increase by too much, and the max should of course also not increase
        if set_weights[min_set]+weights[max_el]-weights[min_el] < set_weights[max_set] &&
            set_weights[max_set]-weights[max_el]+weights[min_el] < set_weights[max_set]
            #swapping decreases the maximum size, so we do it
            sets[max_set] = [i for i in sets[max_set] if i!= max_el]
            push!(sets[max_set],min_el)
            set_weights[max_set] += weights[min_el] - weights[max_el]

            sets[min_set] = [i for i in sets[min_set] if i!= min_el]
            push!(sets[min_set],max_el)
            set_weights[min_set] += weights[max_el] - weights[min_el]
            index_el = 1
            index_set = 1
        elseif index_el < length(sets[index_set]) #some sets have step elements, but others have step-1 elements.
            index_el+=1
        elseif index_el == step-1 && index_set < n-1#try all pairs with the smallest before quitting
            index_set+=1
            index_el = 1
        else
            #nothing changed, so we can as well break
            # println("in the ",k,"th iteration nothing changed anymore so we quit")
            break
        end
    end
    return sets,set_weights,[weights[s] for s in sets]
end

struct BlockInfo
    J::Int # number of constraints
    n_y::Int # number of y variables
    m::Array{Int} #size of constraint polynomial matrices
    L::Array{Int} #number of blocks per constraint
    n_samples::Array{Int} #number of samples per constraint
    Y_blocksizes::Array{Array{Int}} #sizes of the blocks of Y. Not sure if this should be changed to indexing of [j][l] instead
    dim_S::Array{Int} #number of r,s,k tuples per constraint
    x_indices::Array{Int}
    ranks::Array{Array{Array{Int}}} #same size of Y_blocksizes, but then instead of Ints an array of Ints with the ranks for every k.
    rank_sums::Array{Array{Array{Int}}} #the cumulative sum of ranks[j][l][1:k-1]
    nz_k::Array{Array{Int}} # nz_k[j][l] is the first k with nonzero rank for this j,l.
    jl_pairs::Array{Tuple{Int,Int}} #the order in which to process pairs (j,l)
    function BlockInfo(J, n_y, m, L, n_samples, Y_blocksizes, dim_S, ranks)
        #TODO: check for correct combination of L,Y,ranks? -> Y[j], ranks[j] have L[j] blocks
        length(m) == length(L) == length(n_samples) == length(dim_S) == J ||
            error("sizes of m,L,n_samples,dim_S must equal the number of constraints")
        length.(ranks) == length.(Y_blocksizes) == L ||
            error("Y[j] and ranks[j] must have length L[j]")
        x_indices = [sum(dim_S[1:j]) for j = 0:J]
        #Just so that we dont have to calculate this every iteration:
        rank_sums = [[[0, cumsum(ranks[j][l])...] for l = 1:L[j]] for j = 1:J]
        nz_k = [
            [findfirst(k -> ranks[j][l][k] > 0, 1:n_samples[j]) for l = 1:L[j]] for j = 1:J
        ]
        jl_pairs = [(j,l) for j=1:J for l=1:L[j]]
        #the only relevant parameter here is the blocksize Y_blocksizes[j][l]
        #Computing the step length takes a cholesky ~n^3 . Eigenvalues should also take ~n^3
        jl_pair_weights = [Y_blocksizes[j][l]^3 for (j,l) in jl_pairs]
        sets,weights,weight_dist=distribute_weights_swapping(jl_pair_weights,Threads.nthreads(),nswaps = length(jl_pair_weights)^2)
        #This doesnt take much time, and does its job quite well. Of course we cannot get it very even because we just have very large differences in weights
        sort!(sets, by=length, rev=true) #to get these sets on the cores, we need to put the longer sets first
        jl_pairs= jl_pairs[vcat(sets...)]

        new(J, n_y, m, L, n_samples, Y_blocksizes, dim_S, x_indices, ranks, rank_sums, nz_k,jl_pairs)
    end
end
BlockInfo(J, n_y, m, L, n_samples, Y_blocksizes, ranks) = BlockInfo(
    J,
    n_y,
    m,
    L,
    n_samples,
    Y_blocksizes,
    div.(m .* (m .+ 1), 2) .* n_samples,
    ranks,
)

"""Extract the information of BlockInfo for the given constraints"""
function get_block_info(constraints)
    #shouldnt this just be a constructor?
    # i.e. BlockInfo(constraints)
    #constraints = list of [A,B,c,H] (H ≈ A_sign)
    J = length(constraints)

    # B has size #tuples * N
    n_y = size(constraints[1][2], 2)

    #A is indexed by [l,k][r]:
    L = [size(constraints[j][1], 1) for j = 1:J]
    n_samples = [size(constraints[j][1], 2) for j = 1:J]

    # number of tuples = m*(m+1)/2 *n_samples. So m(m+1) = 2*#tuples/n_samples = x
    # which gives m = 1/2(-1+sqrt(4x+1)). Exact, so use isqrt to stay integer
    m = [
        div(-1 + isqrt(8 * div(length(constraints[j][3]), n_samples[j]) + 1), 2) for
        j = 1:J
    ]
    # check if correct:
    @assert all(
        length(constraints[j][3]) == div(m[j] * (m[j] + 1) * n_samples[j], 2) for j = 1:J
    )

    #rank of j,l is the number of vectors in A_j[l,k].
    ranks = [
        [[length(constraints[j][1][l, k]) for k = 1:n_samples[j]] for l = 1:L[j]] for
        j = 1:J
    ]

    #the first k with nonzero rank, to get the blocksize
    nz_rank =
        [[findfirst(k -> ranks[j][l][k] > 0, 1:n_samples[j]) for l = 1:L[j]] for j = 1:J]
    # block sizes of Y is m[j]*length(q_jl)
    Y_blocksizes =
        [[m[j] * length(constraints[j][1][l, nz_rank[j][l]][1]) for l = 1:L[j]] for j = 1:J]

    #rank of j,l is the number of vectors in A_j[l,k].
    ranks = [
        [[length(constraints[j][1][l, k]) for k = 1:n_samples[j]] for l = 1:L[j]] for
        j = 1:J
    ]

    return BlockInfo(J, n_y, m, L, n_samples, Y_blocksizes, ranks)
end

function solvempmp(
    M,
    G,
    q,
    x,
    delta, #same input as prepareabc
    b,
    Pi = nothing;
    all_of_Pi = true,
    kwargs...,
) # Objective vector
    #Get the numerical input for the SDP
    if !isnothing(Pi)
        abc = [
            prepareabc(M[j], G[j], q[j], x[j], delta[j], Pi[j], all_of_Pi = all_of_Pi)
            for j = 1:length(M)
        ]
    else
        abc = [prepareabc(M[j], G[j], q[j], x[j], delta[j]) for j = 1:length(M)]
    end
    # Get the general input about the constraints such as sizes and ranks
    blockinfo = get_block_info(abc)
    # Call the solver
    solverank1sdp(abc, b, blockinfo; kwargs...)
end

#to use C = 0 efficiently. Does not really matter for performance
struct AbsoluteZero end
LinearAlgebra.dot(x::AbsoluteZero, y) = zero(y[1])#gives prec=256 for arbs (default)
Base.:+(X::T, C::AbsoluteZero) where {T} = X
Base.:-(X::T,C::AbsoluteZero) where {T}= X

"""Solve the SDP with rank one constraint matrices."""
function solverank1sdp(
    constraints, # list of (A,B,c,H) tuples (ArbMatrices)
    b, # Objective vector
    blockinfo; # information about the block sizes etc.
    C = 0,
    b0 = 0,
    maxiterations = 500,
    beta_infeasible = T(3) / 10, #try to improve optimality by a factor 1/0.3
    beta_feasible = T(1) / 10, # try to improve optimality by a factor 10
    gamma = T(7) / 10, #what fraction of the maximum step size is used
    omega_p = T(10)^(10), #in general, can be chosen smaller. might need to be increased in some cases
    omega_d = T(10)^(10), # initial variable = omega I
    duality_gap_threshold = T(10)^(-15), # how near to optimal does the solution need to be
    primal_error_threshold = T(10)^(-30),  # how feasible is the primal solution
    dual_error_threshold = T(10)^(-30), # how feasible is the dual solution
    need_primal_feasible = false,
    need_dual_feasible = false,
    testing = true, #Print the times of the first two iterations. This is for testing purposes
    initial_solutions = [],
) # initial solutions of the right format, in the order x,X,y,Y
    #the defaultvalues mostly come from Simmons-Duffin original paper, or from the default values of SDPA-GMP (slow but stable mode)
    #convert to Arbs:
    b = ArbMatrix(b, prec = precision(BigFloat))
    omega_p,
    omega_d,
    gamma,
    beta_feasible,
    beta_infeasible,
    b0,
    duality_gap_threshold,
    primal_error_threshold,
    dual_error_threshold = (
        Arb.(
            [
                omega_p,
                omega_d,
                gamma,
                beta_feasible,
                beta_infeasible,
                b0,
                duality_gap_threshold,
                primal_error_threshold,
                dual_error_threshold,
            ],
            prec = precision(BigFloat),
        )
    )
    #The algorithm:
    #initialize:
    #1): choose initial point q = (0, Ω_p*I, 0, Ω_d*I) = (x,X,y,Y), with Ω>0
    #main loop:
    #2): compute residues  P = ∑_i A_i x_i - X - C, p = b -B^Tx, d = c- Tr(A_* Y) -By and R = μI - XY
    #3): Take μ = Tr(XY)/K, and μ_p = β_p μ with β_p = 0 if q is primal & dual feasible, β_infeasibleible otherwise
    #4): solve system for search direction (dx,dX, dy,dY), with R = μ_p I - XY << most difficult step
    #5): compute corrector deformation μ_c = β_c μ:
    #r = Tr((X+dX)*(Y+dY))/(μK)
    #β = r^2 if r<1, r otherwise
    #β_c = min( max( β_inf, β),1) if primal & dual feasible, max(β_inf,β) otherwise
    #6):solve system for search direction (dx,dX, dy,dY), with R = μ_c I - XY
    #7): compute step lengths: α_p = min(γ α(X,dX),1),α_p = min(γ α(Y,dY),1)
    #with α(M,dM) =  max(0, -eigmin(M)/eigmin(dM)) ( = 0 if eigmin(dM)> 0). Maybe need to do it per block? idk about block matrices etc
    #8): do the steps: x,X -> x,X + α_p dx,dX and y,Y -> y,Y + α_d dy,dY
    #Repeat from step 2

    #step 1: initialize. We may pass a solution from a previous run
    if length(initial_solutions) != 4 #we need the whole solution, x,X,y,Y
        x = ArbMatrix(sum(blockinfo.dim_S), 1, prec = precision(BigFloat)) # all tuples (j,r,s,k), equals size of S.
        X = BlockDiagonal([
            BlockDiagonal([
                ArbMatrix(
                    Matrix{T}(
                        T(omega_p) * I,
                        blockinfo.Y_blocksizes[j][l],
                        blockinfo.Y_blocksizes[j][l],
                    ),
                    prec = precision(BigFloat),
                ) for l = 1:blockinfo.L[j]
            ]) for j = 1:blockinfo.J
        ])
        y = ArbMatrix(blockinfo.n_y, 1, prec = precision(BigFloat))
        Y = BlockDiagonal([
            BlockDiagonal([
                ArbMatrix(
                    Matrix{T}(
                        T(omega_d) * I,
                        blockinfo.Y_blocksizes[j][l],
                        blockinfo.Y_blocksizes[j][l],
                    ),
                    prec = precision(BigFloat),
                ) for l = 1:blockinfo.L[j]
            ]) for j = 1:blockinfo.J
        ])
    else
        #We need to do more checking before using the given starting solutions. After all, these things can be anything.
        (x, X, y, Y) = copy.(initial_solutions)
    end
    if C == 0 #no C objective given. #in principle we can remove C in most cases. Only not for computing residuals
        # AbsoluteZero does nothing when adding or subtracting it to BlockDiagonals
        # In addtion, the dot product with a BlockDiagonal equals 0
        C = AbsoluteZero()# works
    end

    #step 2
    #loop initialization: compute or set the initial values, and print the header
    iter = 1
    @printf(
        "%5s %8s %11s %11s %11s %10s %10s %10s %10s %10s %10s %10s\n",
        "iter",
        "time(s)",
        "μ",
        "P-obj",
        "D-obj",
        "gap",
        "P-error",
        "p-error",
        "d-error",
        "α_p",
        "α_d",
        "beta"
    )
    alpha_p = alpha_d = Arb(0, prec = precision(BigFloat))
    mu = dot(X, Y) / size(X, 1)
    bigfloat_steplength = false #This determines whether the steplength is computed with BigFloats (stabler but slower) or Arblib (faster but less stable due to error bounds)
    spd_inv = true #whether we do the inverse using the spd_inv (faster) or approx_inv (more stable) function from Arblib
    # We overwrite R and X_inv in each iteration
    R = similar(X)
    X_inv = similar(X)
    #errors and feasibility
    p_obj = compute_primal_objective(constraints, x, b0)
    d_obj = compute_dual_objective(y, Y, b, C, b0)
    dual_gap = compute_duality_gap(constraints, x, y, Y, C, b)
    time_res = @elapsed begin
        P, p, d = compute_residuals(constraints, x, X, y, Y, b, C, blockinfo)
    end
    primal_error = compute_primal_error(P, p)
    dual_error = compute_dual_error(d)
    pd_feas = check_pd_feasibility(
        primal_error,
        dual_error,
        primal_error_threshold,
        dual_error_threshold,
    )

    #we keep track of allocations and time of some of the parts of the algorithm
    allocs = zeros(8)
    timings = zeros(17) #timings do not require high precision
    time_start = time()
    @time while (
        !terminate(
            dual_gap,
            primal_error,
            dual_error,
            duality_gap_threshold,
            primal_error_threshold,
            dual_error_threshold,
            need_primal_feasible,
            need_dual_feasible,
        ) && iter < maxiterations
    )
        #step 3
        mu = dot(X, Y) / size(X, 1)
        mu_p = pd_feas ? zero(mu) : beta_infeasible * mu # zero(mu) keeps the precision
        #step 4

        time_R = @elapsed begin
            compute_residual_R!(R, X, Y, mu_p,blockinfo)
        end
        #We invert X per (j,l) block. If one block is close to singularity, spd_inv! may fail due to error bounds; in that case we use approx_inv, which uses an LU decomposition)
        time_inv = @elapsed begin
            Threads.@threads for (j,l) in blockinfo.jl_pairs
                if spd_inv
                    status = Arblib.spd_inv!(
                        X_inv.blocks[j].blocks[l],
                        X.blocks[j].blocks[l],
                    )
                    Arblib.get_mid!(
                        X_inv.blocks[j].blocks[l],
                        X_inv.blocks[j].blocks[l],
                    ) #ignore the error intervals
                    if status == 0
                        Core.println(
                            "The inverse of X could not be computed with the cholesky factorization (with error bounds). We switch to using the LU decomposition (without error bounds).",
                        )
                        # spd_inv went wrong, we use approx_inv from now on, for every block
                        # we can also keep track of the blocks where it went wrong; some blocks keep better conditioning than others (maybe))
                        #NOTE: I haven't ever seen the message yet
                        Arblib.approx_inv!(
                            X_inv.blocks[j].blocks[l],
                            X.blocks[j].blocks[l],
                        )
                        spd_inv = false
                    end
                else
                    succes = Arblib.approx_inv!(
                        X_inv.blocks[j].blocks[l],
                        X.blocks[j].blocks[l],
                    )
                    if succes == 0
                        error(
                            "The inverse was not computed correctly. Try again with higher precision",
                        )
                        #Even the approximate inverse went wrong, so we really need higher precision.
                        #In practice, I do not expect that this will ever happen. It is more likely that the decomposition of S or Q will fail first
                    end
                end
            end
        end
        #Compute the decomposition which is used to solve the system of equations for the search directions. We also keep A_Y, which is used for Tr(A_* Y)
        allocs[1] += @allocated begin
            time_decomp = @elapsed begin
                decomposition, A_Y, time_schur, time_cholS, time_CinvB, time_Q, time_cholQ =
                    compute_T_decomposition(constraints, X_inv, Y, blockinfo)
            end
        end
        # Compute the residuals
        allocs[2] += @allocated begin
            time_res = @elapsed begin
                P, p, d = compute_residuals(constraints, x, X, y, A_Y, b, C, blockinfo)
            end
        end
        #compute the predictor search direction
        allocs[3] += @allocated begin
            time_predictor_dir = @elapsed begin
                dx, dX, dy, dY, times_predictor_in = compute_search_direction(
                    constraints,
                    P,
                    p,
                    d,
                    R,
                    X_inv,
                    Y,
                    blockinfo,
                    decomposition,
                )
            end
        end
        #step 5
        r = dot(X + dX, Y + dY) / (mu * size(X, 1)) # block_diag_dot + generic dot; what about arblib dot?
        beta = r < 1 ? r^2 : r #is arb
        beta_c =
            pd_feas ? min(max(beta_feasible, beta), Arb(1, prec = precision(BigFloat))) :
            max(beta_infeasible, beta)
        mu_c = beta_c * mu

        #step 6
        time_R += @elapsed begin
            compute_residual_R!(R, X, Y, mu_c, dX, dY, blockinfo)
        end
        #Compute the corrector search direction
        allocs[4] += @allocated begin
            time_corrector_dir = @elapsed begin
                dx, dX, dy, dY, times_corrector_in = compute_search_direction(
                    constraints,
                    P,
                    p,
                    d,
                    R,
                    X_inv,
                    Y,
                    blockinfo,
                    decomposition,
                )
            end
        end
        #compute the step lengths
        allocs[5] += @allocated begin
            #step 7
            time_alpha = @elapsed begin
                alpha_p, bigfloat_steplength =
                    compute_step_length(X, dX, gamma, blockinfo,bigfloat_steplength)
                alpha_d, bigfloat_steplength =
                    compute_step_length(Y, dY, gamma, blockinfo,bigfloat_steplength)
            end
        end

        #if the current solution is primal ánd dual feasible, we follow the search direction exactly. (this follows Simmons duffins code)
        if pd_feas
            alpha_p = min(alpha_p, alpha_d)
            alpha_d = alpha_p
        end

        #step 8
        Arblib.addmul!(x, dx, alpha_p)
        Arblib.addmul!(y, dy, alpha_d)
        #we can also do threading over l, which is useful for one/ a few multivariate constraints with symmetry
        #However, this is quadratic time, and the bottleneck is cubic time
        Threads.@threads for (j,l) in blockinfo.jl_pairs
            Arblib.addmul!(X.blocks[j].blocks[l], dX.blocks[j].blocks[l], alpha_p)
            Arblib.get_mid!(X.blocks[j].blocks[l], X.blocks[j].blocks[l])

            Arblib.addmul!(Y.blocks[j].blocks[l], dY.blocks[j].blocks[l], alpha_d)
            Arblib.get_mid!(Y.blocks[j].blocks[l], Y.blocks[j].blocks[l])
        end
        #We save the times of everything except the first 2 iterations, as they may include compile time
        if iter > 2
            timings[1] += time_decomp
            timings[2] += time_predictor_dir
            timings[3] += time_corrector_dir
            timings[4] += time_alpha
            timings[5] += time_inv
            timings[6] += time_R
            timings[7] += time_res
            timings[8:12] .+= [time_schur, time_cholS, time_CinvB, time_Q, time_cholQ]
            timings[13:17] .+= times_predictor_in .+ times_corrector_in
        elseif testing #if testing, the times of the first few iterations may be interesting
            println(
                "decomp:",
                time_decomp,
                ". directions:",
                time_predictor_dir + time_corrector_dir,
                ". steplength:",
                time_alpha,
            )
            println(
                "schur:",
                time_schur,
                " cholS:",
                time_cholS,
                " CinvB:",
                time_CinvB,
                " Q:",
                time_Q,
                " cholQ:",
                time_cholQ,
            )
            println("X inv:", time_inv, ". R:", time_R, ". residuals p,P,d:", time_res)
        end
        #print the objectives of the start of the iteration, imitating simmons duffin
        @printf(
            "%5d %8.1f %11.3e %11.3e %11.3e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e\n",
            iter,
            time() - time_start,
            BigFloat(mu),
            BigFloat(p_obj),
            BigFloat(d_obj),
            BigFloat(dual_gap),
            BigFloat(compute_error(P)),
            BigFloat(compute_error(p)),
            BigFloat(compute_error(d)),
            BigFloat(alpha_p),
            BigFloat(alpha_d),
            beta_c
        )
        #Compute the new objectives and errors, for the new iteration
        allocs[6]+= @allocated begin
        p_obj = compute_primal_objective(constraints, x, b0)
        d_obj = compute_dual_objective(y, Y, b, C, b0)
        dual_gap = compute_duality_gap(p_obj, d_obj)
        primal_error = compute_primal_error(P, p)
        dual_error = compute_dual_error(d)
        end
        #step 2, preparation for new loop iteration
        iter += 1

        pd_feas = check_pd_feasibility(
            primal_error,
            dual_error,
            primal_error_threshold,
            dual_error_threshold)
    end
    time_total = time() - time_start #this may include compile time
    @printf(
        "%5s %8s %11s %11s %11s %10s %10s %10s %10s %10s %10s %10s\n",
        "iter",
        "time(s)",
        "μ",
        "P-obj",
        "D-obj",
        "gap",
        "P-error",
        "p-error",
        "d-error",
        "α_p",
        "α_d",
        "beta"
    )

    #print the total time needed for every part of the algorithm
    println(
        "\nTime spent: (The total time may include compile time. The first few iterations are not included in the rest of the times)",
    )
    @printf(
        "%11s %11s %11s %11s %11s %11s %11s %11s\n",
        "total",
        "Decomp",
        "predict_dir",
        "correct_dir",
        "alpha",
        "Xinv",
        "R",
        "res"
    )
    @printf(
        "%11.5e %11.5e %11.5e %11.5e %11.5e %11.5e %11.5e %11.5e\n\n",
        time_total,
        timings[1:7]...
    )
    println("Time inside decomp:")
    @printf(
        "%11s %11s %11s %11s %11s\n",
        "schur",
        "chol_S",
        "comp CinvB",
        "comp Q",
        "chol_Q"
    )
    @printf("%11.5e %11.5e %11.5e %11.5e %11.5e\n\n", timings[8:12]...)

    println("Time inside search directions (both predictor & corrector step)")
    @printf(
        "%11s %11s %11s %11s %11s\n",
        "calc Z",
        "calc rhs x",
        "solve system",
        "calc dX",
        "calc dY"
    )
    @printf("%11.5e %11.5e %11.5e %11.5e %11.5e\n\n", timings[13:end]...)
    println(allocs)
    return x,
    X,
    y,
    Y,
    P,
    p,
    d,
    compute_duality_gap(constraints, x, y, Y, C, b),
    compute_primal_objective(constraints, x, b0),
    compute_dual_objective(y, Y, b, C, b0),
    time_total
end
"""Compute the primal objective <c, x> + b0"""
function compute_primal_objective(constraints, x, b0)
    return dot_c(constraints, x) + b0
end

"""Compute the dual objective <C,Y> + <b,y> + b0"""
function compute_dual_objective(y, Y, b, C, b0)
    return dot(C, Y) + dot(b, y) + b0 # block_diag_dot
end

"""Compute the error (max abs (P_ij)) of a blockdiagonal matrix"""
function compute_error(P::BlockDiagonal)
    max_P = Arb(0, prec = precision(BigFloat))
    for b in P.blocks
        Arblib.max!(max_P, compute_error(b), max_P)
    end
    return max_P
end
"""Compute the error (max abs(d_ij)) of a matrix"""
function compute_error(d::ArbMatrix)
    max_d = Arb(0, prec = precision(d))
    abs_temp = Arb(0, prec = precision(d))
    for i = 1:size(d, 1)
        for j = 1:size(d, 2)
            Arblib.max!(max_d, max_d, Arblib.abs!(abs_temp, ref(d, i, j)))
        end
    end
    return max_d
    # return max(abs.(d)...)
end

"""Compute the primal error"""
function compute_primal_error(P, p)
    max_p = compute_error(p)
    max_P = compute_error(P)
    return Arblib.max!(max_p, max_p, max_P)
end
"""Compute the dual error"""
compute_dual_error(d) = compute_error(d)

"""Compute the duality gap"""
function compute_duality_gap(constraints, x, y, Y, C, b)
    primal_objective = dot_c(constraints, x)
    dual_objective = dot(C, Y) + dot(b, y) # block_diag_dot, dot
    duality_gap =
        abs(primal_objective - dual_objective) /
        max(one(primal_objective), abs(primal_objective + dual_objective))
    return duality_gap
end
function compute_duality_gap(primal_objective, dual_objective)
    return abs(primal_objective - dual_objective) /
           max(one(primal_objective), abs(primal_objective + dual_objective))
end

"""Compute <c,x> where c is distributed over constraints"""
function dot_c(constraints, x)
    res = Arb(0, prec = precision(x))
    x_idx = 1
    #We do this manually to avoid allocations
    for j = 1:length(constraints)
        for i = 1:length(constraints[j][3])
            Arblib.addmul!(res, ref(constraints[j][3], i, 1), ref(x, x_idx, 1))
            x_idx += 1
        end
    end
    return res
end

"""Compute the dual residue d = c- Tr(A_* Y) - By"""
function calculate_res_d(constraints,y,Y,blockinfo)
    #vcat gives Any back according to @code_warntype, so we explicitely say that this is an ArbMatrix
    d::ArbMatrix = vcat([constraints[j][3] for j=1:blockinfo.J]...)
    B::ArbMatrix = vcat([constraints[j][2] for j=1:blockinfo.J]...) ##or we can do  B[j]y and then vcat
    Arblib.sub!(d,d,Arblib.approx_mul!(similar(d),B,y))
    Arblib.sub!(d,d,trace_A(constraints,Y,blockinfo))
    return d
end



"""Compute the residuals P,p and d."""
function compute_residuals(constraints, x, X, y, Y, b, C, blockinfo)
    # P = ∑_i A_i x_i - X - C,
    P = similar(X)
    #P = ∑_i x_i A_i
    compute_weighted_A!(P, constraints, x, blockinfo)
    #P-= X
    #P-=C (if necessary)
    Threads.@threads for (j,l) in blockinfo.jl_pairs
        Arblib.sub!(P.blocks[j].blocks[l],P.blocks[j].blocks[l],X.blocks[j].blocks[l])
        if typeof(C) != AbsoluteZero
            Arblib.sub!(P.blocks[j].blocks[l],P.blocks[j].blocks[l], C.blocks[j].blocks[l])
        end
        Arblib.get_mid!(P.blocks[j].blocks[l], P.blocks[j].blocks[l])
    end

    # d = c- Tr(A_* Y) -By
    d = calculate_res_d(constraints,y,Y,blockinfo)
    Arblib.get_mid!(d, d)

    #p = b-B^T x
    p = similar(b)
    #We do it per thread separately, but they need to be added together which cannot directly be done with threading
    p_added = [zero(p) for j = 1:blockinfo.J]
    Threads.@threads for j = 1:blockinfo.J
        cur_x = 1
        j_idx = sum(blockinfo.dim_S[1:j-1])
        B_transpose = ArbMatrix(blockinfo.n_y,blockinfo.dim_S[j],prec=precision(BigFloat))
        Arblib.transpose!(B_transpose,constraints[j][2])
        Arblib.approx_mul!(p_added[j],B_transpose, x[j_idx+1:j_idx+blockinfo.dim_S[j],:])
    end # end of j
    for j=1:blockinfo.J
        Arblib.sub!(p,p,p_added[j])
    end
    Arblib.add!(p,p,b)
    Arblib.get_mid!(p, p)

    return P, p, d
end

"""Determine whether the main loop should terminate or not"""
function terminate(
    duality_gap,
    primal_error,
    dual_error,
    duality_gap_threshold,
    primal_error_threshold,
    dual_error_threshold,
    need_primal_feasible,
    need_dual_feasible,
)
    duality_gap_opt = duality_gap < duality_gap_threshold
    primal_feas = primal_error < primal_error_threshold
    dual_feas = dual_error < dual_error_threshold
    if need_primal_feasible && primal_feas
        println("Primal feasible solution found")
        return true
    end
    if need_dual_feasible && dual_feas
        println("Dual feasible solution found")
        return true
    end
    if primal_feas && dual_feas && duality_gap_opt
        println("Optimal solution found")
        return true
    end
    return false
end

"""Check primal and dual feasibility"""
function check_pd_feasibility(
    primal_error,
    dual_error,
    primal_error_threshold,
    dual_error_threshold,
)
    primal_feas = primal_error < primal_error_threshold
    dual_feas = dual_error < dual_error_threshold
    return primal_feas && dual_feas
end


"""Compute the residual R, with or without second order term """
function compute_residual_R!(R, X, Y, mu,blockinfo)
    # R = mu*I - XY
    Threads.@threads for (j,l) in blockinfo.jl_pairs
        Arblib.one!(R.blocks[j].blocks[l])
        Arblib.mul!(R.blocks[j].blocks[l],R.blocks[j].blocks[l],mu)
        XY = similar(R.blocks[j].blocks[l])
        Arblib.approx_mul!(XY,
            X.blocks[j].blocks[l],
            Y.blocks[j].blocks[l])
        Arblib.sub!(R.blocks[j].blocks[l], R.blocks[j].blocks[l], XY)
    end
    return R
end

function compute_residual_R!(R, X, Y, mu, dX, dY,blockinfo)
    # R = mu*I - XY -dXdY
    Threads.@threads for (j,l) in blockinfo.jl_pairs
        Arblib.one!(R.blocks[j].blocks[l])
        Arblib.mul!(R.blocks[j].blocks[l],R.blocks[j].blocks[l],mu)
        temp = similar(R.blocks[j].blocks[l])
        Arblib.approx_mul!(temp,X.blocks[j].blocks[l],Y.blocks[j].blocks[l])
        Arblib.sub!(R.blocks[j].blocks[l], R.blocks[j].blocks[l], temp)
        Arblib.approx_mul!(temp, dX.blocks[j].blocks[l], dY.blocks[j].blocks[l])
        Arblib.sub!(R.blocks[j].blocks[l], R.blocks[j].blocks[l], temp)
    end
    return R
end

"""Compute S, integrated with the precomputing of the bilinear pairings"""
function compute_S_integrated(constraints, X_inv, Y, blockinfo)
    S = [
        ArbMatrix(blockinfo.dim_S[j], blockinfo.dim_S[j], prec = precision(BigFloat))
        for j = 1:blockinfo.J
    ]
    #bilinear pairings are only used per j,l. So we can compute them per j,l, and then make S_j.
    #For Tr(A_* Y) we need bilinear_pairings_Y[r,s,k,rnk,k,rnk]. So per r,s block, the diagonal
    A_Y = [
        [Matrix{ArbMatrix}(undef, blockinfo.m[j], blockinfo.m[j]) for l = 1:blockinfo.L[j]] for j = 1:blockinfo.J
    ]
    for j = 1:blockinfo.J
        for l = 1:blockinfo.L[j]
            # #Assuming we use all threads. We can give an option in the solver which says how much threads may be used max.
            used_threads = Threads.nthreads()
            #if we use this amount of threads, we need either floor(amount/used_threads) or ceil(amount/used_threads) number of vectors
            rank_sums = blockinfo.rank_sums[j][l] #the cumulative sum of the number of vectors up to sample k
            min_thread_size = div(rank_sums[end], used_threads)
            # We assign amount_min_thread_size or amount_min_thread_size+1 to each thread
            amount_min_thread_size =
                (min_thread_size + 1) * used_threads -
                sum(blockinfo.ranks[j][l][k] for k = 1:blockinfo.n_samples[j])
            thread_sizes = vcat(
                [min_thread_size + 1 for i = 1:used_threads-amount_min_thread_size],
                [min_thread_size for i = 1:amount_min_thread_size],
            )
            indices = [0, cumsum(thread_sizes)...]

            # initialize
            # We concatenate all vectors to a matrix
            #Because hcat and vcat somehow seem to be type-instable, we say that vectors is an ArbMatrix.
            # The type instability probably occurs when n_samples[j] = 0 or ranks[j][l][k] = 0 for all k
            vectors::ArbMatrix = hcat(
                [
                    constraints[j][1][l, k][rnk] for k = 1:blockinfo.n_samples[j] for
                    rnk = 1:blockinfo.ranks[j][l][k]
                ]...,
            )
            # delta is the length of 1 vector
            delta = size(vectors, 1)
            # block_size is the the total number of vectors, the size of each block (r,s)
            block_size = size(vectors, 2)
            vectors_trans = ArbMatrix(block_size, delta, prec = precision(BigFloat))
            Arblib.transpose!(vectors_trans, vectors)
            #initialize the result matrices
            bilinear_pairings_Xinv = ArbMatrix(blockinfo.m[j] * block_size,
                blockinfo.m[j] * block_size,prec = precision(BigFloat))
            bilinear_pairings_Y = ArbMatrix(blockinfo.m[j] * block_size,
                blockinfo.m[j] * block_size, prec = precision(BigFloat))

            part_matrix_Xinv = ArbMatrix(blockinfo.m[j] * delta,
                size(vectors, 2),prec = precision(BigFloat))
            part_matrix_Y = ArbMatrix(blockinfo.m[j] * delta,
                size(vectors, 2),prec = precision(BigFloat))

            #compute the bilinear pairings: (V ⊗ I)^T Y (V ⊗ I)
            # instead of computing V ⊗ I, we compute the parts corresponding to different r,s manually
            Threads.@threads for i = 1:used_threads
                #Compute the block of bilinear pairings corresponding to indices[i]+1:indices[i+1
                cur_part = vectors[:, indices[i]+1:indices[i+1]]

                #initialize the matrices which are used as scratch space
                #(you cannot directly write to an indexed part of the matrix by approx_mul!)
                #(Indexing creates a copy, so you write to a anonymous copy)
                cur_part_block_Xinv = ArbMatrix(blockinfo.m[j] * delta,
                    indices[i+1] - indices[i],prec = precision(BigFloat))
                cur_part_block_Y = ArbMatrix(blockinfo.m[j] * delta,
                    indices[i+1] - indices[i],prec = precision(BigFloat))

                Xinv_part = ArbMatrix(block_size,indices[i+1] - indices[i],
                    prec = precision(BigFloat))
                Y_part = ArbMatrix(block_size,indices[i+1] - indices[i],
                    prec = precision(BigFloat))
                for s = 1:blockinfo.m[j]
                    Arblib.approx_mul!(cur_part_block_Xinv,
                        X_inv.blocks[j].blocks[l][:, (s-1)*delta+1:s*delta],
                        cur_part)
                    Arblib.approx_mul!(cur_part_block_Y,
                        Y.blocks[j].blocks[l][:, (s-1)*delta+1:s*delta],
                        cur_part)

                    for r = 1:blockinfo.m[j]
                        #V^T * (X^-1 V)
                        Arblib.approx_mul!(Xinv_part,
                            vectors_trans,
                            cur_part_block_Xinv[(r-1)*delta+1:r*delta, :])
                        bilinear_pairings_Xinv[
                            (r-1)*block_size+1:r*block_size,
                            (s-1)*block_size+1+indices[i]:(s-1)*block_size+indices[i+1],
                        ] = Xinv_part
                        #V^T * (YV)
                        Arblib.approx_mul!(Y_part,
                            vectors_trans,
                            cur_part_block_Y[(r-1)*delta+1:r*delta, :],
                        )
                        bilinear_pairings_Y[
                            (r-1)*block_size+1:r*block_size,
                            (s-1)*block_size+1+indices[i]:(s-1)*block_size+indices[i+1],
                        ] = Y_part
                    end
                end
            end
            #We collect the parts v^T Y v, because we need them to compute Tr(A_* Y)
            for r = 1:blockinfo.m[j]
                for s = 1:blockinfo.m[j]
                    A_Y[j][l][r, s] =
                        ArbMatrix(rank_sums[end], 1, prec = precision(BigFloat))
                    for k = 1:rank_sums[end]
                        index1 = (r - 1) * block_size + k
                        index2 = (s - 1) * block_size + k
                        A_Y[j][l][r, s][k, 1] = ref(bilinear_pairings_Y, index1, index2)
                    end
                end
            end


            # We compute the contribution of this l to S[j].
            # For a different k1, we add to a different element of S[j], so threading is allowed
            Threads.@threads for k1 = 1:blockinfo.n_samples[j]
                tot = Arb(0, prec = precision(BigFloat))
                for r1 = 1:blockinfo.m[j]
                    for s1 = 1:r1
                        #index for the tuple (r1,s1,k1):
                        hor_el =
                            k1 + ((s1 - 1) + div(r1 * (r1 - 1), 2)) * blockinfo.n_samples[j]
                        for r2 = 1:blockinfo.m[j]
                            for s2 = 1:r2
                                for k2 = 1:blockinfo.n_samples[j]
                                    #index for the tuple (r2,s2,k2):
                                    ver_el = k2 +
                                        ((s2 - 1) + div(r2 * (r2 - 1), 2)) *
                                        blockinfo.n_samples[j]
                                    if ver_el <= hor_el #upper triangular part
                                        for rnk1 = 1:blockinfo.ranks[j][l][k1],
                                            rnk2 = 1:blockinfo.ranks[j][l][k2]
                                            #calculate the entries of the bilinear pairing matrices corresponding to r1,s1,r2 and s2
                                            #Outer block is r,s. Each block has size sum(ranks[k] for k=1:n_samples), so to get in the right block we need that size *(r-1)
                                            #In the inner block, we first have all the ranks for k=1:k-1. Then we need element rnk after that
                                            r1_spot =
                                                rnk1 + #innermost block
                                                rank_sums[k1] + #ranks in this r, for other k
                                                rank_sums[end] * (r1 - 1) #ranks for different r
                                            r2_spot =
                                                rnk2 +
                                                rank_sums[k2] +
                                                rank_sums[end] * (r2 - 1)
                                            s1_spot =
                                                rnk1 +
                                                rank_sums[k1] +
                                                rank_sums[end] * (s1 - 1)
                                            s2_spot =
                                                rnk2 +
                                                rank_sums[k2] +
                                                rank_sums[end] * (s2 - 1)

                                            #we overwrite tot in the first muL!.
                                            Arblib.mul!(
                                                tot,
                                                ref(bilinear_pairings_Xinv,s1_spot,r2_spot),
                                                ref(bilinear_pairings_Y, s2_spot, r1_spot),
                                            )
                                            Arblib.addmul!(
                                                tot,
                                                ref(bilinear_pairings_Xinv,r1_spot,r2_spot),
                                                ref(bilinear_pairings_Y, s2_spot, s1_spot),
                                            )
                                            Arblib.addmul!(
                                                tot,
                                                ref(bilinear_pairings_Xinv,s1_spot,s2_spot),
                                                ref(bilinear_pairings_Y, r2_spot, r1_spot),
                                            )
                                            Arblib.addmul!(
                                                tot,
                                                ref(bilinear_pairings_Xinv,r1_spot,s2_spot),
                                                ref(bilinear_pairings_Y, r2_spot, s1_spot),
                                            )
                                            #constants in front of the sum:
                                            Arblib.mul!(tot,constraints[j][4][l, k1][rnk1],tot)
                                            Arblib.mul!(tot,constraints[j][4][l, k2][rnk2],tot)
                                            Arblib.div!(tot, tot, 4)
                                            #add the result to the corresponding element of S[j]
                                            S[j][ver_el, hor_el] = Arblib.add!(tot,tot,ref(S[j], ver_el, hor_el))
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end

        end
        S[j] .= Symmetric(S[j]) #symmetrize
        Arblib.get_mid!(S[j], S[j])
    end

    return S, A_Y
end

"""Compute the decomposition of [S B; B^T 0]"""
function compute_T_decomposition(constraints, X_inv, Y, blockinfo)
    # 1) pre compute bilinear basis (uses constraints[j][1],X^{-1},Y)
    # 2) compute S (uses precomputed bilinear products)
    # 3) compute cholesky decomposition S =CC^T (uses S)
    # 4) compute decomposition of [S B; B^T 0] (uses constraints[j][2],S)

    # 1,2) compute the bilinear pairings and S, integrated. (per j,l, compute first pairings then S[j] part)
    time_schur = @elapsed begin
        #this gives the most allocations
        S, A_Y = compute_S_integrated(constraints, X_inv, Y, blockinfo)
    end

    #3) cholesky decomposition of S
    #NOTE: Instead of the (instable) cholesky decomposition, we use the approx_lu! from Arblib
    #  and then we use the decomposition (L 0; B^TU^-1 I)(I 0; 0 B^TU^-1L^-1B) (U -L^-1B; 0 I)
    # The Cholesky decomposition is instable due to the error bounds, but there is no approximate version in Arblib. Converting to BigFloats gives extra allocations and is (a lot) slower than approx_lu (for large sizes)
    time_cholS = @elapsed begin
        perms = [zeros(Int, size(S[j], 1)) for j = 1:blockinfo.J]
        Threads.@threads for j = 1:blockinfo.J
            succes = Arblib.approx_lu!(perms[j], S[j], S[j], prec = precision(BigFloat))
            perms[j] .+= 1
            if succes == 0
                error("S was not decomposed succesfully, try again with higher precision")
            end
        end
    end

    #4) compute decomposition:
    #L^-1B and B^T U^-1
    time_CinvB = @elapsed begin
        LinvB = [
            ArbMatrix(blockinfo.dim_S[j], blockinfo.n_y, prec = precision(BigFloat)) for j = 1:blockinfo.J
        ]
        BTUinv = [
            ArbMatrix(blockinfo.n_y, blockinfo.dim_S[j], prec = precision(BigFloat)) for j = 1:blockinfo.J
        ]

        Threads.@threads for j = 1:blockinfo.J
            #We take the permutation into account by permuting the rows of B (L^-1P^-1B)
            #We use LinvB as scratch space for U^-TB -> B^T U^-1
            C_trans = similar(S[j])
            Arblib.transpose!(C_trans, S[j])
            Arblib.approx_solve_tril!(LinvB[j], C_trans, constraints[j][2], 0) #we use U, so including diagonal
            Arblib.transpose!(BTUinv[j], LinvB[j])
            #compute L^-1B; we overwrite the used scratch space
            # This allocates because we use ArbMatrix[...,:]
            Arblib.approx_solve_tril!(LinvB[j], S[j], constraints[j][2][perms[j], :], 1)
        end
    end
    #compute Q = B^T U^-1 (PL)^-1 B
    time_Q = @elapsed begin
        #We distribute the matrices about equally over several threads, and then sum the results
        total_LinvB::ArbMatrix = vcat(LinvB...)
        total_BTUinv::ArbMatrix = hcat(BTUinv...)
        used_threads = Threads.nthreads()
        min_size = div(size(total_LinvB, 1), used_threads)
        amount_min_thread_size = (min_size + 1) * used_threads - size(total_LinvB, 1)
        thread_sizes = vcat(
            [min_size + 1 for i = 1:used_threads-amount_min_thread_size],
            [min_size for i = 1:amount_min_thread_size],
        )
        indices = [0, cumsum(thread_sizes)...]

        #one result for every thread
        Q = [
            ArbMatrix(blockinfo.n_y, blockinfo.n_y, prec = precision(BigFloat)) for
            i = 1:used_threads
        ]

        Threads.@threads for i = 1:used_threads
            #allocations here because ArbMatrix[.:.,.:.] creates a copy.
            Arblib.approx_mul!(
                Q[i],
                total_BTUinv[:, indices[i]+1:indices[i+1]],
                total_LinvB[indices[i]+1:indices[i+1], :],
            )
        end
        Q_summed::ArbMatrix = sum(Q)
    end

    # compute the LU factors of Q (instead of cholesky, for stability)
    #Without error bounds
    time_cholQ = @elapsed begin
        perm = zeros(Int64,size(Q_summed, 1))
        succes = Arblib.approx_lu!(perm, Q_summed, Q_summed, prec = precision(Q_summed))
        if succes == 0
            error("Q was not decomposed correctly. Try restarting with a higher precision.")
        end
    end

    return (S, perms, LinvB, BTUinv, perm, Q_summed), #the blocks that
    A_Y,
    time_schur,
    time_cholS,
    time_CinvB,
    time_Q,
    time_cholQ
end

"""Compute the vector Tr(A_* Z) for one or all constraints"""
function trace_A(constraints, Z::BlockDiagonal, blockinfo)
    #Assumption: Z is symmetric
    result = ArbMatrix(sum(blockinfo.dim_S), 1, prec = precision(BigFloat))
    #result has one entry for each (j,r,s,k) tuple
    for j = 1:blockinfo.J
        j_idx = sum(blockinfo.dim_S[1:j-1])
        for l = 1:blockinfo.L[j]
            used_threads = Threads.nthreads()
            #if we use this amount of threads, we need either floor(amount/used_threads) or ceil(amount/used_threads) number of samples per thread
            rank_sums = blockinfo.rank_sums[j][l]#we use this for calcuatling the
            min_thread_size = div(rank_sums[end], used_threads)
            #Here we make sure that we distribute all samples exactly once over the threads
            amount_min_thread_size =(min_thread_size + 1) * used_threads - rank_sums[end]
            thread_sizes = vcat(
                [min_thread_size + 1 for i = 1:used_threads-amount_min_thread_size],
                [min_thread_size for i = 1:amount_min_thread_size],
            )
            indices = [0, cumsum(thread_sizes)...]
            for r=1:blockinfo.m[j]
                for s=1:r
                    #Approach of Simmons Duffin: (Z[r,s] V) ∘ V (where ∘ is the entrywise (hadamard) product, and V is the matrix of all vectors as columns
                    # Then we need to add this to the block of result corresponding to j
                    #However, we need to take care of different ranks. So we can calculate it like this, but we cannot put the result just in the vector because we have low rank instead of rank 1
                    # time_cat += @elapsed begin
                    #     allocs_cat+= @allocated begin
                    vs::ArbMatrix = hcat([constraints[j][1][l,k][rnk] for k=1:blockinfo.n_samples[j] for rnk=1:blockinfo.ranks[j][l][k]]...)
                #     end
                # end
                    # vs_scaled = hcat([constraints[j][4][l,k][rnk]*constraints[j][1][l,k][rnk] for k=1:blockinfo.n_samples[j] for rnk=1:blockinfo.ranks[j][l][k]]...)
                    # vs_transpose_scaled = ArbMatrix(size(vs,2),size(vs,1),prec=precision(BigFloat))
                    # Arblib.transpose!(vs_transpose_scaled,vs_scaled)
                    vs_transpose = ArbMatrix(size(vs,2),size(vs,1),prec=precision(BigFloat))
                    Arblib.transpose!(vs_transpose,vs)
                    delta = size(vs,1)
                    result_parts = [ArbMatrix(indices[i+1]-indices[i],1,prec=precision(BigFloat)) for i=1:used_threads]
                    ones = ArbMatrix(delta,1,prec=precision(BigFloat))
                    Arblib.ones!(ones)

                    Threads.@threads for i=1:used_threads
                        vs_transpose_part = vs_transpose[indices[i]+1:indices[i+1],:]
                        VZ = ArbMatrix(indices[i+1]-indices[i],delta,prec=precision(BigFloat))
                        Arblib.approx_mul!(VZ,vs_transpose_part,Z.blocks[j].blocks[l][(r-1)*delta+1:r*delta, (s-1)*delta+1:s*delta]) # we can parallellize here over the samples (rows of vs_transpose)
                        Arblib.mul_entrywise!(VZ,VZ,vs_transpose_part)
                        Arblib.approx_mul!(result_parts[i],VZ,ones)

                    end
                    res_offset = j_idx+ ((s-1)+div((r-1)*r,2))*blockinfo.n_samples[j]
                    result_part_summed = ArbMatrix(blockinfo.n_samples[j],1,prec=precision(BigFloat))
                    result_part::ArbMatrix = vcat(result_parts...)
                    #we don't have to vcat them, we can also keep track of the i_idx in the summing part. However, this is easier and (probably) doesnt cost much
                    #now we add the right parts to the right entries, multiplied by the eigenvalues
                    res = Arb(prec=precision(BigFloat))
                    idx = 1
                    for k=1:blockinfo.n_samples[j]
                        Arblib.zero!(res)
                        for rnk=1:blockinfo.ranks[j][l][k]
                            Arblib.addmul!(res,constraints[j][4][l,k][rnk],ref(result_part,idx,1))
                            idx+=1
                        end
                        Arblib.add!(res,res,ref(result,res_offset+k,1))
                        result[res_offset+k,1] = res
                    end
                end
            end
        end
    end
    return result
end
function trace_A(constraints, A_Y, blockinfo)
    #NOTE: here we have precomputed v^T A v already
    #Assumption: Z is symmetric. Tr(Z λvv^T 1/2(e_r e_s + e_s e_r)) =1/2 λ(Tr(Z[r,s]vv^T +Z[s,r]vv^T)) = λv^TZ[r,s]v
    result = ArbMatrix(sum(blockinfo.dim_S), 1, prec = precision(BigFloat))
    #we can parallellize over the samples because we calculate the index
    for j=1:blockinfo.J
        j_idx = sum(blockinfo.dim_S[1:j-1])
        Threads.@threads for k = 1:blockinfo.n_samples[j]
            for l = 1:blockinfo.L[j]
                rank_sums = blockinfo.rank_sums[j][l]
                for rnk = 1:blockinfo.ranks[j][l][k]
                    #Because we have v^TYv already, we just have to multiply with the right constants and add it to the right entry
                    res = Arb(0, prec = precision(BigFloat))
                    for r = 1:blockinfo.m[j]
                        for s = 1:r
                            #calculate the index corresponding to this j,r,s,k (in result)
                            tup_idx = k + blockinfo.n_samples[j] * ((s - 1) + div(r * (r - 1), 2))+j_idx
                            #calculate the index in the A_Y matrix
                            idx = rnk + rank_sums[k]
                            Arblib.mul!(
                                res,
                                constraints[j][4][l, k][rnk],
                                ref(A_Y[j][l][r, s], idx, 1),
                            )
                            Arblib.add!(res, res, ref(result, tup_idx, 1))
                            result[tup_idx, 1] = res
                        end
                    end
                end
            end
        end
    end
    return result
end

"""Set initial_matrix to ∑_i a_i A_i"""
function compute_weighted_A!(initial_matrix, constraints, a, blockinfo)
    #initial matrix is block matrix of block matrices of ArbMatrices
    #NOTE: instead of adding Q to both the r,s block and the s,r block, we can add it to the upper block and use Symmetric()
    for j = 1:blockinfo.J
        j_idx = sum(blockinfo.dim_S[1:j-1])
        for l = 1:blockinfo.L[j]
            #The first k with nonzero rank. If it doesnt exist, then this l does not belong to this constraint.
            nz_k = blockinfo.nz_k[j][l]
            delta = length(constraints[j][1][l, nz_k][1])

            #threading only depends on the outer part of the matrix, in this case that is the length of the vectors (delta)
            #We calculate the start and end indices for each thread. This does not depend on r,s
            used_threads = Threads.nthreads()
            min_thread_size = div(delta, used_threads)
            amount_min_thread_size = (min_thread_size + 1) * used_threads - delta
            thread_sizes = vcat(
                [min_thread_size + 1 for i = 1:used_threads-amount_min_thread_size],
                [min_thread_size for i = 1:amount_min_thread_size],
            )
            indices = [0, cumsum(thread_sizes)...]
            for r = 1:blockinfo.m[j]
                for s = 1:r
                    #Approach: sum_i a_i A_i can be written as V_i D V_i^T, with D diagonal (λ_i,rnk * a_i)
                    #NOTE: Do we ever need vs as vectors? Otherwise it might save time & allocations if we just save vs instead of the vectors separately (in general)
                    #Here we can either parallellize over the samples (columns of V_i), so a part  of the sum
                    # or over the basis (rows of V_i), so obtaining a part of the final matrix.
                    #We choose for the second approach; in that case we do not have to sum the results at the end.
                    #However, in general the length of the basis is (factor >=2) less than the number of constraints in a cluster. So the other method might use the threads in a better way.
                    #Initialize the matrices, both V^T and DV
                    vs::ArbMatrix = hcat([constraints[j][1][l,k][rnk] for k=1:blockinfo.n_samples[j] for rnk=1:blockinfo.ranks[j][l][k]]...)
                    vs_transpose = ArbMatrix(blockinfo.rank_sums[j][l][end], delta, prec = precision(BigFloat))
                    Arblib.transpose!(vs_transpose,vs)
                    v_offset = (s - 1 + div(r * (r - 1), 2)) * blockinfo.n_samples[j]+j_idx
                    vs_scaled::ArbMatrix = hcat([ref(a,k + v_offset,1)*constraints[j][4][l,k][rnk]*constraints[j][1][l,k][rnk] for k=1:blockinfo.n_samples[j] for rnk=1:blockinfo.ranks[j][l][k]]...)

                    Threads.@threads for i=1:used_threads
                        #calculate VD * V^T
                        Q_part = ArbMatrix(delta,indices[i+1]-indices[i],prec=precision(BigFloat))
                        Arblib.approx_mul!(Q_part,vs_scaled,vs_transpose[:,indices[i]+1:indices[i+1]])
                        #On offdiagonal blocks, we have the factor 1/2 from E_rs (=1/2* e_r e_s^T + e_s e_r^T)
                        if r != s
                            Arblib.div!(Q_part,Q_part, 2)
                        end
                        initial_matrix.blocks[j].blocks[l][
                            (s-1)*delta+1:s*delta,
                            (r-1)*delta+1+indices[i]:(r-1)*delta + indices[i+1],
                        ] = Q_part
                    end
                end
            end
            if blockinfo.m[j] != 1
                #We only have to symmetrize when there are offdiagonal blocks (i.e. r!=s)
                initial_matrix.blocks[j].blocks[l] .= Symmetric(initial_matrix.blocks[j].blocks[l])
            end
        end
    end
    return nothing #initial matrix modified and return nothing
end


"""Compute the search directions, using a precomputed decomposition"""
function compute_search_direction(
    constraints,
    P,
    p,
    d,
    R,
    X_inv,
    Y,
    blockinfo,
    (C, perms, LinvB, BTUinv, perm, Q),
)
    # using the decomposition, compute the search directions
    # 5) solve system with rhs dx,dy <- (-d - Tr(A_* Z) ;  p) with Z = X^{-1}(PY - R)
    # 6) compute dX = P + sum_i A_i dx_i
    # 7) compute dY = X^{-1}(R-dX Y) (XdY = R-dXY)
    # 8) symmetrize dY = 1/2 (dY +dY')
    time_Z = @elapsed begin
        Z = similar(Y)
        Threads.@threads for (j,l) in blockinfo.jl_pairs
            #Z = X_inv*(P*Y-R)
            #We use Z as scratch space; Z = X_inv ((PY)-R)
            # for l = 1:blockinfo.L[j]
            Arblib.approx_mul!(
                Z.blocks[j].blocks[l],
                P.blocks[j].blocks[l],
                Y.blocks[j].blocks[l],
            )
            Arblib.sub!(
                Z.blocks[j].blocks[l],
                Z.blocks[j].blocks[l],
                R.blocks[j].blocks[l],
            )
            Arblib.approx_mul!(
                Z.blocks[j].blocks[l],
                X_inv.blocks[j].blocks[l],
                Z.blocks[j].blocks[l],
            )
            #We symmetrize Z in order to use Tr(A_*Z) correctly
            temp_t = similar(Z.blocks[j].blocks[l])
            Arblib.transpose!(temp_t, Z.blocks[j].blocks[l])
            Arblib.add!(temp_t, Z.blocks[j].blocks[l], temp_t)
            Arblib.mul!(
                Z.blocks[j].blocks[l],
                temp_t,
                Arb(1 // 2, prec = precision(BigFloat)),
            )
            Arblib.get_mid!(Z.blocks[j].blocks[l], Z.blocks[j].blocks[l])
        end
    end
    #the right hand sides of the system
    rhs_y = p
    time_rhs_x = @elapsed begin
        #rhs_x = -d-Tr(A_*Z)
        rhs_x = similar(d)
        Arblib.neg!(rhs_x,d)
        Arblib.sub!(rhs_x,rhs_x,trace_A(constraints, Z, blockinfo)) #this one allocates
        Arblib.get_mid!(rhs_x, rhs_x)
    end

    # solve the system (C 0; B^TU^-1 I)(I 0; 0 Q_LQ_U)(C^T -L^-1B; 0 I)(dx; dy) = (rhs_x; rhs_y)
    indices = blockinfo.x_indices #0, dim_S[1], dim_S[1]+dim_S[2],... ,sum(dim_S)
    time_sys = @elapsed begin
        #The first lower triangular system: Cx = rhs_x
        temp_x = [
            ArbMatrix(indices[j+1] - indices[j], 1, prec = precision(BigFloat)) for
            j = 1:blockinfo.J
        ]
        temp_y = [similar(rhs_y) for j = 1:blockinfo.J]
        #y = rhs_y - B^TU^-1 x
        Threads.@threads for j = 1:blockinfo.J
                Arblib.approx_solve_tril!(
                temp_x[j],
                C[j],
                rhs_x[perms[j] .+ indices[j], :],
                1,
            ) #lower has diagonal 1
            Arblib.approx_mul!(temp_y[j], BTUinv[j], temp_x[j]) #now its an ArbMatrix, otherwise ArbVector which uses generic multipilcations
        end
        dy = similar(rhs_y)
        Arblib.sub!(dy, rhs_y, sum(temp_y))
        # dy = rhs_y - sum(temp_y)
        #second system: temp_x stays the same, dy_new  =  Q^-1 dy_old
        Arblib.approx_solve_lu_precomp!(dy, perm, Q, dy)

        #third system: dy stays the same, Udx = dx' + LinvB dy
        dx_perj = [
            ArbMatrix(indices[j+1] - indices[j], 1, prec = precision(BigFloat)) for
            j = 1:blockinfo.J
        ]
        Threads.@threads for j = 1:blockinfo.J
            Arblib.approx_solve_triu!(dx_perj[j], C[j], (temp_x[j] + LinvB[j] * dy), 0)
        end
        dx::ArbMatrix = vcat([dx_perj[j] for j = 1:blockinfo.J]...)
        Arblib.get_mid!(dx, dx) #not sure if this is needed, because we used approx_solve_triu! (i.e. the approx version)
    end #of timing system

    #step 6:
    time_dX = @elapsed begin
        dX = similar(P) # making sure dX does not reference to P anymore
        #dX = ∑_i dx_i A_i + P
        compute_weighted_A!(dX, constraints, dx, blockinfo)
        Threads.@threads for (j,l) in blockinfo.jl_pairs
            Arblib.add!(dX.blocks[j].blocks[l],dX.blocks[j].blocks[l],P.blocks[j].blocks[l])
        end
    end

    #step 7 & 8: compute dY and symmetrize
    time_dY = @elapsed begin
        dY = similar(Y)
        Threads.@threads for (j,l) in blockinfo.jl_pairs
            #dY = X_inv * (R- dX *Y) = X_inv * ( R- (dX*Y))
            #Again, we use dY as scratch space
            # for l = 1:blockinfo.L[j]
            Arblib.approx_mul!(
                dY.blocks[j].blocks[l],
                dX.blocks[j].blocks[l],
                Y.blocks[j].blocks[l],
            )
            Arblib.sub!(
                dY.blocks[j].blocks[l],
                R.blocks[j].blocks[l],
                dY.blocks[j].blocks[l],
            )
            Arblib.approx_mul!(
                dY.blocks[j].blocks[l],
                X_inv.blocks[j].blocks[l],
                dY.blocks[j].blocks[l],
            )
            #Symmetrize dY
            temp_t = similar(dY.blocks[j].blocks[l])
            Arblib.transpose!(temp_t, dY.blocks[j].blocks[l])
            Arblib.mul!(
                dY.blocks[j].blocks[l],
                dY.blocks[j].blocks[l] + temp_t,
                Arb(1 // 2, prec = precision(BigFloat)),
            )
            Arblib.get_mid!(dY.blocks[j].blocks[l], dY.blocks[j].blocks[l])
            Arblib.get_mid!(dX.blocks[j].blocks[l], dX.blocks[j].blocks[l])
        end
    end

    return dx, dX, dy, dY, [time_Z, time_rhs_x, time_sys, time_dX, time_dY]
end


"""Compute the step length min(γ α(M,dM), 1), where α is the maximum number step
to which keeps M+α(M,dM) dM positive semidefinite"""
function compute_step_length(
    M::BlockDiagonal,
    dM::BlockDiagonal,
    gamma,
    blockinfo,
    bigfloat_steplength = false,
)
    print_switch = !bigfloat_steplength
    # We parallellize over j, but need the total minimum. We keep track of a minimum for every j
    # We actually just need to keep track of the minimum for every thread.
    # min_eig_bf = [[T(Inf) for l=1:length(M.blocks[j].blocks)] for j = 1:length(M.blocks)]
    min_eig_arb = [[Arb(Inf) for l=1:length(M.blocks[j].blocks)] for j = 1:length(M.blocks)]
    try #We catch the errors for the (BigFloat) cholesky, to give more information.
        Threads.@threads for (j,l) in blockinfo.jl_pairs
            if bigfloat_steplength == false
                #We first try it with Arb until that does not work anymore
                chol = similar(M.blocks[j].blocks[l])
                succes = Arblib.cho!(chol, M.blocks[j].blocks[l]) #might be
                if succes == 0
                    bigfloat_steplength = true #switch to bigFloat computations
                else
                    Arblib.get_mid!(chol,chol)
                    LML = similar(dM.blocks[j].blocks[l])
                    #LML = chol^-1 dMblock
                    Arblib.approx_solve_tril!(LML, chol, dM.blocks[j].blocks[l], 0)
                    Arblib.transpose!(LML, LML)
                    #temp LML = chol^-1 (chol^-1 dMblock)^T
                    Arblib.approx_solve_tril!(LML, chol, LML, 0)
                    eigenvalues = AcbVector(size(LML, 1))
                    #converting to AcbMatrix allocates, but we cannot avoid it because this function is only available for AcbMatrices
                    # We need to do it somewhere, so we can as well use the faster Arb stuff for the previous operations
                    succes2 = Arblib.approx_eig_qr!(eigenvalues, AcbMatrix(LML))
                    if succes2 == 0 #even in this case it is possible that the output is accurate enough (arblib docs)
                        bigfloat_steplength = true
                    else
                        real_arb = Arb(0, prec = precision(BigFloat))
                        for i = 1:length(eigenvalues)
                            #We get zero imaginary part (if it is correct)
                            #NOTE: Do we need to check that the imaginary part is approximately zero?
                            Arblib.get_real!(real_arb, ref(eigenvalues, i))
                            Arblib.min!(min_eig_arb[j][l], real_arb, min_eig_arb[j][l])
                        end
                    end
                end
            end
            if bigfloat_steplength == true
                #This is the backup if the cholesky from Arblib does not work due to error bounds
                cholb = cholesky(T.(M.blocks[j].blocks[l]))
                LMLb = cholb.L \ T.(dM.blocks[j].blocks[l]) / cholb.U
                min_eig_arb[j][l] = min(eigmin((LMLb + LMLb') / 2), min_eig_arb[j][l])
            end
        end
    catch e
        error("The step length could not be calculated correctly. Try a higher precision.")
        # throw(e)
    end
    if print_switch && bigfloat_steplength
        printnl("The computation of the steplength is switching to BigFloat calculations due to instabilities.")
    end

    #We need the minimum eigenvalue
    min_eig_J = [min(BigFloat.(min_eig_arb[j])...) for j=1:length(M.blocks)]
    min_eig = min(min_eig_J...)

    if min_eig > -gamma
        return Arb(1, prec = precision(BigFloat)), bigfloat_steplength
    else
        return Arb(-gamma / min_eig, prec = precision(BigFloat)), bigfloat_steplength
    end
end


end #of module MPMP
