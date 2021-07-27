module MPMP

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
function make_monomial_basis(poly_ring,d)
    #NOTE: this is the monomial basis, so in general a very bad choice
    #works for any number of variables
    n = nvars(poly_ring)
    vars = gens(poly_ring)
    q = zeros(poly_ring,binomial(n+d,d))#n+d choose d basis polynomials
    q_idx = 1
    for k = 0:d
        for exponent in multiexponents(n,k) #exponent vectors of size n with total degree k.
            temp_pol = MPolyBuildCtx(poly_ring)
            push_term!(temp_pol,poly_ring(1)(zeros(n)...),exponent)
            temp_pol = finish(temp_pol)
            q[q_idx] = temp_pol
            q_idx+=1
        end
    end
    return q
end

function laguerrebasis(k::Integer, alpha, x)
    #Davids function
    v = Vector{typeof(one(alpha)*one(x))}(undef, 1+k)
    v[1] = one(x)
    k == 0 && return v
    v[2] = 1 + alpha - x
    k == 1 && return v
    for l = 2:k
        v[l+1] = 1//big(l) * ((2l-1+alpha-x) * v[l] - (l+alpha-1) * v[l-1])
    end
    return v
end

function jacobi_basis(d::Integer,alpha,beta,x,normalized=true)
    q = Vector{typeof(one(alpha)*one(x))}(undef,d+1)
    q[1] = one(x)
    d == 0 && return q
    q[2] = x # normalized
    if !normalized
        q[2] *=  (alpha + 1)
    end
    d == 1 && return q
    for k=2:d
        #what if alpha+beta = -n for some integer n>=1
        q[k+1] = (2*k+alpha+beta-1)/BigFloat(2k*(k+alpha+beta)*(2k+alpha+beta-2))*((2*k+alpha+beta)*(2k+alpha+beta-2)*x+beta^2-alpha^2)*q[k]+
                -2*(k+alpha-1)*(k+beta-1)*(2*k+alpha+beta)*q[k-1]
        # q[k+1] = (2*k+alpha+ beta-1)*(k+alpha)/BigFloat(k*(k+2*alpha))*x*q[k] - (k+alpha-1)*(k+alpha)/BigFloat(k*(k+2*alpha))*q[k-1]
    end
    return q
end

function gegenbauer_basis(d::Integer, alpha, x, normalized = true)
    #construct a basis of Gegenbauer C^alpha_k polynomials up to degree d (inclusive). #from wikipedia
    # for alpha = (n/2-1) this is the P^n_k(x) used. orthogonal wrt the measure (1-x^2)^{alpha-1/2}dx.
    q = Vector{typeof(one(BigFloat(alpha))*one(x))}(undef, 1+d)
    q[1] = one(x)
    d == 0 && return q
    q[2] = x # normalized s.t. pol(1) =1 for all polynomials (recurrence preserves pol_k(1) = 1)
    if !normalized
        q[2] *= 2 * alpha
    end
    d == 1 && return q
    for k=2:d #q[3] gives degree k=2
        q[k+1] = 2*(k+alpha-1)/BigFloat(k)*x*q[k] - (k+2*alpha-2)/BigFloat(k)*q[k-1]
    end
    return q
end

function create_sample_points(n,d)
    #rational points in the unit simplex with denominator d
    #probably not very efficient, but I dont know how to do it better for general n.
    x = [zeros(BigFloat,n) for i=1:binomial(n+d,d)] #need n+d choose d points for a unisolvent set, if symmetry is not used.
    idx = 1
    for I in CartesianIndices(ntuple(k->0:d,n)) #all tuples with elements in 0:d of length n
        if sum(Tuple(I))<=d #in unit simplex
            x[idx] = [i/BigFloat(d) for i in Tuple(I)]
            idx+=1
        end
    end
    return x
end

function create_sample_points_2d(d)
    #padua points:
    z = [Array{BigFloat}(undef,2) for i=1:binomial(2+d,d)]
    z_idx = 1
    for j = 0:d
        delta_j = j%2 == d%2 == 1 ? 1 : 0
        mu_j = cospi(j/d)
        for k=1:(div(d,2)+1 + delta_j)
            eta_k = j%2 == 1 ? cospi((2*k-2)/(d+1)) : cospi((2*k-1)/(d+1))
            z[z_idx] = [mu_j,eta_k]
            z_idx+=1
        end
    end
    return z
end

function create_sample_points_3d(d; pairs = [(1,3),(3,2),(2,1)])#of pairs tested was this the best one. Good for odd n. This approach does not really work for even n.
    d%2 == 0 && println("n should be odd for the sample points to be good. Consider using different sample points.")
    # extension of padua & chebyshev points. Similar to how padua points are an extension of chebyshev points
    pad = create_sample_points_2d(d) #size (n+1)*(n+2)/2
    pad_div = [pad[1:3:end],pad[2:3:end],pad[3:3:end]] # should we divide it up in a different way?
    ch = create_sample_points_chebyshev(d+2) #size (n+3)
    cheb_div = [ch[1:3:end],ch[2:3:end],ch[3:3:end]]
    total_points = [similar([pad[1]...,ch[1]]) for i=1:div((d+1)*(d+2)*(d+3),6)]
    cur_point = 1
    for pair in pairs
        for p1 in pad_div[pair[1]]
            for p2 in cheb_div[pair[2]]
                total_points[cur_point] =[p1..., p2]
                cur_point+=1
            end
        end
    end
    return total_points
end

function points_X_general(n,d)# sometimes good, not always.
    #works with n=4: d=2,3,5,11. Not for d=4,6,7,8
    if n==2
        return MPMP.create_sample_points_2d(d)
    end
    Xn_1 = points_X_general(n-1,d)
    cheb = MPMP.create_sample_points_chebyshev(d+n-1);println(length(Xn_1));println(length(cheb))
    X_div = [Xn_1[i:n:end] for i=1:n]
    cheb_div = [cheb[i:n:end] for i=1:n]
    total_points = [similar([Xn_1[1]...,cheb[1]]) for i=1:binomial(n+d,d)]
    cur_point = 1
    for i=1:n
        j = i==1 ? n : i-1
        for p1 in X_div[i]
            for p2 in cheb_div[j]
                total_points[cur_point] = [p1...,p2]
                cur_point+=1
            end
        end
    end
    return total_points
end



# function create_sample_points_3d_try2(d)
#     n=3
#     chebs = [create_sample_points_chebyshev(d+i) for i=0:n-1] #sizes d+1...d+n
#     chebs_div = [[chebs[1][i:6:end] for i=1:6], [chebs[2][i:6:end] for i=1:6], [chebs[i:3:end] for i=1:3]]
#     #combine first two sets
#     temp = chebs_div[1]

function create_sample_points_1d(d)
    #as done in simmons duffin: ('rescaled Laguerre')
    # x[k] = sqrt(pi)/(64*(d+1)*log( 3- 2*sqrt(2))) * (-1+4*k)^2, with k=0:d
    constant = -sqrt(BigFloat(pi))/(64*(d+1)*log(3-2*sqrt(BigFloat(2))))
    x = zeros(BigFloat,d+1)
    for k = 0:d
        x[k+1] = constant*(-1+4*k)^2
    end
    return x
end

function create_sample_points_chebyshev(d,a=-1,b=1)
    #roots of chebyshev polynomials of the first kind, unisolvent for polynomials up to degree d
    return [(a+b)/BigFloat(2) + (b-a)/BigFloat(2) * cos((2k-1)/BigFloat(2(d+1))*BigFloat(pi)) for k=1:d+1]
end

function create_sample_points_chebyshev_mod(d,a=-1,b=1)
    #roots of chebyshev polynomials of the first kind, divided by cos(pi/2(d+1)) to get a lower lebesgue constant
    return [(a+b)/BigFloat(2) + (b-a)/BigFloat(2) * cos((2k-1)/BigFloat(2(d+1))*BigFloat(pi))/cos(BigFloat(pi)/2(d+1)) for k=1:d+1]
end

## Functions for the solver
#cholesky with Arblib:
# include("Arblib_functions.jl")


#extending LinearAlgebra.dot for our BlockDiagonal matrices. in principle 'type piracy'
function LinearAlgebra.dot(A::BlockDiagonal, B::BlockDiagonal)
   #assume that A and B have the same blockstructure
   sum(dot(a,b) for (a,b) in zip(blocks(A), blocks(B)))
end

#extending max and abs to easily use max(abs(B)) for a blockdiagonal efficiently
#not used atm
function max(A::AbstractMatrix)
    return max(A...)
end
function max(B::BlockDiagonal)
    return max(max(b) for b in B.blocks)
end
function abs(A::AbstractMatrix)
    return abs.(A)
end
function abs(B::BlockDiagonal)
    return BlockDiagonal(abs.(B.blocks))
end

#NOTE: currently the format is g qq^T ⊗ Pi . The eigenvalues of Pi and the sign of g are stored together.
#       Now that we have the eigenvalues of Pi also in A_sign, we can just as well put the whole G there.
# other possibility is to make the structure qq^T ⊗ Pi, with g included in Pi
function prepareabc(M, # Vector of matrix polynomials
                    G, # Vector of polynomials
                    q, # Vector of polynomials
                    x, # Vector of points
                    δ = -1,# maximum degree. negative numbers-> use 2* the total degree of q[end]
                    Pi = nothing;
                    prec = precision(BigFloat))  #polynomial matrices, as much as G has. We will use A_(jrsk) = sum_l Tr(Q ⊗ E_rs) where Q = ∑_η λ_η(x_k)(√G[l](x_k)q(x_k) ⊗ v_η(x_k)) (√G[l](x_k)q(x_k) ⊗ v_η(x_k))^T
                    #So λ_η sign(G[l](x_k)) is stored in A_sign[l,k][η] = H_(l,k,η), and the vectors (√G[l](x_k)q(x_k) ⊗ v_η(x_k)) in A[l,k][η]
    # Assumptions:
    # 1) all elements of M are the same size (true for a constraint)
    # 2) all matrices in this constraint are constructed from the same polynomial ring (i.e M[i][r,s] all have the same number of variables)
    # 3) the first k polynomials of q span the space of n-variate totaldegree d polynomials, where k+1 is the first index with a higher degree polynomial
    # 4) (if no max degree given) q contains precisely enough polynomomials to span the whole space (last polynomial has degree δ/2)

    m = size(M[1], 1)
    n = nvars(parent(M[1][1,1])) #the number of variables of the polynomial ring of M[1][1,1].
    if δ<0
        δ = 2*total_degree(q[end]) #assumption: q is exactly long enough
    end

    if isnothing(Pi)
        Pi_vecs = [[[T(1)]] for l in G,k=1:length(x)]
        Pi_vals = [[T(1)] for l in G,k=1:length(x)]
        deg_Pi = [0 for l in G]
    else
        #use SVD to get eigenvalues and vectors. getting eigenvectors for bigfloatmatrices is not possible with eigen(M) or eigvecs(M)
        svd_decomps = [svd([Pi[l][i,j](x[k]...) for i=1:size(Pi[l],1), j=1:size(Pi[l],2)]) for l=1:length(G), k=1:length(x)]
        Pi_vecs = [ [svd_decomps[l,k].U[:,r] for r=1:size(Pi[l],1)] for l=1:length(G), k=1:length(x)]
        Pi_vals = [ [sign(dot(svd_decomps[l,k].U[:,r],svd_decomps[l,k].Vt[:,r]))*svd_decomps[l,k].S[r] for r=1:size(Pi[l],1)] for l=1:length(G), k=1:length(x)]
        deg_Pi = [max([total_degree(Pi[l][i,j]) for i=1:size(Pi[l],1) for j=1:size(Pi[l],2)]...) for l=1:length(G)]
        #extra check: sum(pi_vals* pi_vecs*pi_vecs^T) = pi
        max_dif = 0
        index = [0,0,0,0]
        for k=1:length(x)
            for l=1:length(G)
                for i=1:size(Pi[l],1)
                    for j=1:size(Pi[l],2)
                        #test whether we correctly decomposed Pi[l][i,j][x[k]...]
                        cur_dif = Pi[l][i,j](x[k]...)
                        for r=1:size(Pi[l],1)
                            cur_dif -= Pi_vecs[l,k][r][i]*Pi_vecs[l,k][r][j]*Pi_vals[l,k][r]
                        end
                        if abs(cur_dif)>10^(-10)
                            println([Pi[l][i,j](x[k]...) for i=1:size(Pi[l],1),j=1:size(Pi[l],1)])
                            println(cur_dif)
                            println(k,l,i,j)
                        end
                        if abs(cur_dif)>max_dif
                            max_dif = abs(cur_dif)
                            index = [k,l,i,j]
                        end
                    end
                end
            end
        end
        if max_dif>0
            println(max_dif,index)
        end
    end

    # find the last occurance of a degree in the basis.
    # Needed for symmetries, where the number of required basis polynomials can be (much) smaller than (n+d choose d)
    degrees = ones(Int64,div(δ,2)+1) #maximum degree needed is δ/2. everything is an index, so at least 1
    cur_deg = 0
    all_degrees = [total_degree(q[i]) for i=1:length(q)]
    #check for monotonicity:
    for i=1:length(all_degrees)-1
        if all_degrees[i] > all_degrees[i+1]
            println("Degrees are not monotone. The program will (most probably) not be correct if you don't fix this")
        end
    end
    last_deg = [findlast(x->x==i,all_degrees) for i=0:div(δ,2)] #at place d+1: the last last index i such that deg(q[i]) = d
    #change the nothings into the previous one
    for i=1:length(last_deg)
        if isnothing(last_deg[i])
            last_deg[i] = last_deg[i-1] #always works if 1 is the first entry and the degrees are monotone
        end
    end
    # for i = 1:length(q)
    #     if total_degree(q[i]) == cur_deg +1
    #         #the polynomial at place i has larger degree;
    #         #we havent increased the cur_degree yet so this is the first occurance
    #         # hence i-1 is the last entry with polynomial of degree cur_deg
    #         degrees[cur_deg+1] = i-1
    #         cur_deg+=1
    #         if cur_deg+1 > length(degrees)
    #             #we only need degrees up to div(δ,2)
    #             #allowing cur_deg+1>length(degrees) would also raise an index error
    #             break
    #         end
    #     elseif total_degree(q[i])<cur_deg || total_degree(q[i])>cur_deg+1
    #         #degree can be cur_degree (still basis for prev part) or cur_degree+1 (start of basis for new part)
    #         println("Something might be wrong, the degree of q is not monotone increasing with steps of 1")
    #     end
    #     if i == length(q) && total_degree(q[i]) == cur_deg #at the end, with maximum degree
    #         degrees[cur_deg+1] = i
    #     end
    # end
    # println(degrees)
    # We can even put the whole G[l](x[k]...) in the 'sign'. We already have the eigenvalues of the Pi there
    # either way, better to be consistent (either also G in A_sign, or only the signs -> ev of Pi in A)
    A_sign = [[Arb(Pi_vals[l,k][r]*sign(G[l](x[k]...)),prec=prec) for r=1:length(Pi_vals[l,k])] for l=1:length(G),k=1:length(x)]
    A = [ [ArbMatrix(kron([q[d](x[k]...)*sqrt(abs(G[l](x[k]...))) for d=1:last_deg[div(δ-total_degree(G[l])-deg_Pi[l],2)+1]],Pi_vecs[l,k][r]),prec=prec) for r=1:length(Pi_vecs[l,k])] for l=1:length(G), k=1:length(x)]
    # println(degrees,q)
    # println(length(A[1,1][1]))
    # println(div(δ,2))
    # println(div(δ-total_degree(G[1])-deg_Pi[1],2)+1)
    # A[l,k][r] is the vector v_{j,l,k,r} for this constraint j
    # A_sign[l,k][r] gives the sign and the r'th eigenvalue of Pi. i.e Q = \sum_r A_sign[l,k][r] A[l,k][r] *A[l,k][r]'

    B = ArbMatrix(vcat([transpose([T(-M[i][r,s](x[k]...)) for i=2:length(M)]) for r=1:m for s=1:r for k=1:length(x)]...),prec=prec)

    c = ArbMatrix(vcat([[T(M[1][r,s](x[k]...))] for r=1:m for s=1:r for k=1:length(x)]...),prec=prec)
    @assert precision(B) == prec
    @assert precision(c) == prec
    # println(size(B))
    return A, B, c, A_sign
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
    ranks::Array{Array{Int}} #same size of Y_blocksizes
    function BlockInfo(J,n_y,m,L,n_samples,Y_blocksizes,dim_S,ranks)
        #TODO: check for correct combination of L,Y,ranks? -> Y[j], ranks[j] have L[j] blocks
        length(m)==length(L)==length(n_samples)==length(dim_S) == J || error("sizes of m,L,n_samples,dim_S must equal the number of constraints")
        length.(ranks) == length.(Y_blocksizes) == L || error("Y[j] and ranks[j] must have length L[j]")
        x_indices = [sum(dim_S[1:j]) for j=0:J]
        new(J,n_y,m,L,n_samples,Y_blocksizes,dim_S,x_indices,ranks)
    end
end
BlockInfo(J,n_y,m,L,n_samples,Y_blocksizes,ranks) = BlockInfo(J,n_y,m,L,n_samples,Y_blocksizes, div.(m.*(m.+1),2) .*n_samples,ranks)

"""Extract the information of BlockInfo for the given constraints"""
function get_block_info(constraints)
    #constraints = list of [A,B,c,H] (H ≈ A_sign)
    J = length(constraints)

    # B has size #tuples * N
    n_y = size(constraints[1][2],2)

    #A is indexed by [l,k][r]:
    L = [size(constraints[j][1],1) for j=1:J]
    n_samples = [size(constraints[j][1],2) for j=1:J]

    # number of tuples = m*(m+1)/2 *n_samples. So m(m+1) = 2*#tuples/n_samples = x
    # which gives m = 1/2(-1+sqrt(4x+1)). Exact, so use isqrt to stay integer
    m = [div(-1+isqrt(8*div(length(constraints[j][3]),n_samples[j])+1),2) for j=1:J]
    # check if correct:
    @assert all(length(constraints[j][3]) == div(m[j]*(m[j]+1)*n_samples[j],2) for j=1:J)

    # block sizes of Y is m[j]*length(q_jl)
    Y_blocksizes = [[m[j]*length(constraints[j][1][l,1][1]) for l=1:L[j]] for j=1:J]

    #rank of j,l is the number of vectors in A_j[l,k]
    ranks = [[length(constraints[j][1][l,1]) for l=1:L[j]] for j=1:J]

    return BlockInfo(J,n_y,m,L,n_samples,Y_blocksizes,ranks)
end

function solvempmp(M,G,q,x,delta, #same input as prepareabc
                   b,Pi=nothing; kwargs...) # Objective vector
    if !isnothing(Pi)
        abc = [prepareabc(M[j],G[j],q[j],x[j],delta[j],Pi[j]) for j=1:length(M)]
    else
        abc = [prepareabc(M[j],G[j],q[j],x[j],delta[j]) for j=1:length(M)]
    end
    # Use prepareabc to call solverank1sdp.
    blockinfo = get_block_info(abc)
    solverank1sdp(abc, b, blockinfo; kwargs...)
end

#to use C = 0 efficiently. Does not really matter for performance
struct AbsoluteZero end
LinearAlgebra.dot(x::AbsoluteZero,y) = eltype(y)(0)
Base.:+(X::BlockDiagonal, C::AbsoluteZero) = X
LinearAlgebra.dot(x::ArbMatrix,y::ArbMatrix) = Arblib.approx_dot!(Arb(0,prec=precision(x)),Arb(0,prec=precision(x)),0,x[:],1,y[:],1,length(x))

"""Solve the SDP with rank one constraint matrices."""
function solverank1sdp(constraints, # list of (A,B,c,H) tuples (ArbMatrices)
                       b, # Objective vector
                       blockinfo; # information about the block sizes etc.
                       C=0, b0 = 0, maxiterations=150,
                       beta_infeas = T(3)/10, #try to improve optimality by a factor 1/0.3
                       beta_feas= T(1)/10, # try to improve optimality by a factor 10
                       gamma =  T(7)/10, #what fraction of the maximum step size is used
                       omega_p = T(10)^(10), #in general, can be chosen smaller. might need to be increased in some cases
                       omega_d = T(10)^(10), # initial variable = omega I
                       duality_gap_threshold = T(10)^(-15), # how near to optimal does the solution need to be
                       primal_error_threshold = T(10)^(-30),  # how feasible is the primal solution
                       dual_error_threshold = T(10)^(-30), # how feasible is the dual solution
                       need_primal_feasible = false,
                       need_dual_feasible = false,
                       testing = false,
                       initial_solutions = []) # initial solutions of the right format, in the order x,X,y,Y
                       #the defaultvalues mostly come from Simmons-Duffin original paper
    #convert to Arbs:
    b = ArbMatrix(b,prec=precision(BigFloat))
    omega_p,omega_d,gamma,beta_feas,beta_infeas,b0,duality_gap_threshold,primal_error_threshold,dual_error_threshold = (
    Arb.([omega_p,omega_d,gamma,beta_feas,beta_infeas,b0,duality_gap_threshold,primal_error_threshold,dual_error_threshold],prec=precision(BigFloat)))

    #initialize:
    #1): choose initial point q = (0, Ω_p*I, 0, Ω_d*I) = (x,X,y,Y), with Ω>0
    #main loop:
    #2): compute residues  P = ∑_i A_i x_i - X - C, p = b -B^Tx, d = c- Tr(A_* Y) -By and R = μI - XY
    #3): Take μ = Tr(XY)/K, and μ_p = β_p μ with β_p = 0 if q is primal & dual feasible, β_infeasible otherwise
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

    #step 1: initialize.
    if length(initial_solutions) != 4 #we need the whole solution, x,X,y,Y
        x = ArbMatrix(sum(blockinfo.dim_S),1,prec=precision(BigFloat)) # all tuples (j,r,s,k), equals size of S.
        X = BlockDiagonal([BlockDiagonal([ArbMatrix(Matrix{T}(T(omega_p)*I,blockinfo.Y_blocksizes[j][l],blockinfo.Y_blocksizes[j][l]),prec=precision(BigFloat)) for l=1:blockinfo.L[j]]) for j=1:blockinfo.J])
        y = ArbMatrix(blockinfo.n_y,1,prec=precision(BigFloat)) #first bigfloat then arbmatrix. Not that clean but okay
        Y = BlockDiagonal([BlockDiagonal([ArbMatrix(Matrix{T}(T(omega_d)*I,blockinfo.Y_blocksizes[j][l],blockinfo.Y_blocksizes[j][l]),prec=precision(BigFloat)) for l=1:blockinfo.L[j]]) for j=1:blockinfo.J])
    else
        (x,X,y,Y) = copy.(initial_solutions)
    end
    if C == 0 #no C objective given. #in principle we can remove C in most cases. Only for computing residuals; not sure on the impact regarding memory (as it is always zero)
        # C = zero(Y) #make sure adding/subtracting C works in case it is 0.
        C = AbsoluteZero() # works
    end

    #step 2
    time_res = @elapsed begin
        P,p,d = compute_residuals(constraints,x,X,y,Y,b,C,blockinfo)
    end
    #loop initialization
    iter = 1
    @printf("%5s %8s %11s %11s %11s %10s %10s %10s %10s %10s %10s %10s\n", "iter","time(s)","μ", "P-obj","D-obj","gap","P-error","p-error","d-error","α_p","α_d","beta")
    alpha_p = alpha_d = Arb(0,prec=precision(BigFloat))
    mu = dot(X,Y)/size(X,1) #block_diag_dot; NOTE: dot is not done in Arb; they dont have a dot product for matrices
    p_obj = compute_primal_objective(constraints,x,b0)
    d_obj = compute_dual_objective(y,Y,b,C,b0)
    dual_gap = compute_duality_gap(constraints,x,y,Y,C,b)
    primal_error = compute_primal_error(P,p)
    dual_error = compute_dual_error(d)
    pd_feas = check_pd_feasibility(primal_error,dual_error,primal_error_threshold,dual_error_threshold)
    spd_inv = true #whether we do the inverse using the spd_inv (faster) or approx_inv (more stable) function from Arblib

    timings = zeros(Float64,17) #timings do not require high precision
    time_start = time()
    while (!terminate(dual_gap,primal_error,dual_error,duality_gap_threshold,primal_error_threshold,dual_error_threshold,need_primal_feasible,need_dual_feasible)
        && iter < maxiterations)
        #step 3
        mu = dot(X,Y)/size(X,1) # block_diag_dot,
        mu_p =  pd_feas ? zero(mu) : beta_infeas * mu # zero(mu) keeps the precision

        #step 4
        time_R = @elapsed begin
            R = compute_residual_R(X,Y,mu_p)
        end
        time_inv = @elapsed begin
            X_inv = similar(X)
            Threads.@threads for j=1:blockinfo.J
                for l=1:blockinfo.L[j]
                    if spd_inv
                        status = Arblib.spd_inv!(X_inv.blocks[j].blocks[l],X.blocks[j].blocks[l])
                        Arblib.get_mid!(X_inv.blocks[j].blocks[l],X_inv.blocks[j].blocks[l]) #ignore the error intervals, not needed for approx_inv
                        if status == 0
                            # spd_inv went wrong, we use approx_inv from now on, for every block
                            # we can also keep track of the blocks where it went wrong; some blocks keep better conditioning than others (probably)
                            Arblib.approx_inv!(X_inv.blocks[j].blocks[l],X.blocks[j].blocks[l])
                            spd_inv = false
                        end
                    else
                        Arblib.approx_inv!(X_inv.blocks[j].blocks[l],X.blocks[j].blocks[l])
                    end
                end
            end
        end

        time_decomp = @elapsed begin
            decomposition, time_schur,time_cholS,time_CinvB, time_Q,time_cholQ = compute_T_decomposition(constraints,X_inv,Y,blockinfo)
        end

        time_predictor_dir = @elapsed begin
            dx,dX,dy,dY,times_predictor_in = compute_search_direction(constraints,P,p,d,R,X_inv,Y,blockinfo,decomposition)
        end

        #step 5
        r = dot(X+dX,Y+dY)/(mu*size(X,1)) # block_diag_dot + generic dot; what about arblib dot?
        beta = r<1 ? r^2 : r #is arb
        beta_c = pd_feas ? min(max(beta_feas,beta),Arb(1,prec=precision(BigFloat))) : max(beta_infeas, beta)
        mu_c = beta_c*mu

        #step 6
        time_R += @elapsed begin
            R = compute_residual_R(X,Y,mu_c,dX,dY) #overwrite R for less memory usage? Maybe do it inplace? we have R already
        end

        time_corrector_dir = @elapsed begin
            dx,dX,dy,dY,times_corrector_in = compute_search_direction(constraints,P,p,d,R,X_inv,Y,blockinfo,decomposition)
        end

        #step 7
        time_alpha = @elapsed begin
            alpha_p = compute_step_length(X,dX,gamma)
            alpha_d = compute_step_length(Y,dY,gamma)
        end

        #if primal & dual feasible, follow search direction exactly. (this follows Simmons duffins code)
        if pd_feas
            alpha_p = min(alpha_p,alpha_d)
            alpha_d = alpha_p
        end
        #step 8
        Arblib.addmul!(x,dx,alpha_p)
        Arblib.addmul!(y,dy,alpha_d)
        # println("update dX dY")
        Threads.@threads for j=1:blockinfo.J
            Arblib.addmul!.(X.blocks[j].blocks,dX.blocks[j].blocks,alpha_p)
            Arblib.get_mid!.(X.blocks[j].blocks,X.blocks[j].blocks)

            Arblib.addmul!.(Y.blocks[j].blocks,dY.blocks[j].blocks,alpha_d)
            Arblib.get_mid!.(Y.blocks[j].blocks,Y.blocks[j].blocks)
        end

        # dx = dX = dy = dY = nothing #does this help in memory usage?
        if iter > 2 # do not save times of first few iterations, they can include compile times
            timings[1] += time_decomp
            timings[2] += time_predictor_dir
            timings[3] += time_corrector_dir
            timings[4] += time_alpha
            timings[5] += time_inv
            timings[6] += time_R
            timings[7] += time_res
            timings[8:12] .+= [ time_schur,time_cholS,time_CinvB,time_Q,time_cholQ]
            timings[13:17] .+= times_predictor_in .+ times_corrector_in
        else #if testing #if testing, the times of the first few iterations may be interesting
            println("decomp:",time_decomp,". directions:",time_predictor_dir+time_corrector_dir,". steplength:",time_alpha)
            println("schur:",time_schur," cholS:", time_cholS, " CinvB:", time_CinvB," Q:",time_Q," cholQ:",time_cholQ)
            println("X inv:",time_inv,". R:", time_R,". residuals p,P,d:",time_res)
        end
        #print the objectves of the start of the iteration, imitating simmons duffin
        @printf("%5d %8.1f %11.3e %11.3e %11.3e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e\n",iter,time()-time_start,BigFloat(mu),BigFloat(p_obj),BigFloat(d_obj),BigFloat(dual_gap),BigFloat(max([max(abs.(p)...) for p in P.blocks]...)),BigFloat(max(abs.(p)...)),BigFloat(max(abs.(d)...)),BigFloat(alpha_p),BigFloat(alpha_d),beta_c)

        p_obj = compute_primal_objective(constraints,x,b0)
        d_obj = compute_dual_objective(y,Y,b,C,b0)
        dual_gap = compute_duality_gap(constraints,x,y,Y,C,b)
        primal_error = compute_primal_error(P,p)
        dual_error = compute_dual_error(d)

        #step 2, preparation for new loop iteration
        iter+=1
        time_res = @elapsed begin
            P,p,d= compute_residuals(constraints,x,X,y,Y,b,C,blockinfo)
        end
        pd_feas = check_pd_feasibility(primal_error,dual_error,primal_error_threshold,dual_error_threshold) #are we computing things two times? i.e. here and in terminate?
    end
    time_total = time()-time_start #this may include compile time
    @printf("%5s %8s %11s %11s %11s %10s %10s %10s %10s %10s %10s %10s\n", "iter","time(s)","μ", "P-obj","D-obj","gap","P-error","p-error","d-error","α_p","α_d","beta")

    #print the total time needed for every part of the algorithm
    println("\nTime spent: (The total time may include compile time. The first few iterations are not included in the rest of the times)")
    @printf("%11s %11s %11s %11s %11s %11s %11s %11s\n","total","Decomp","predict_dir","correct_dir","alpha","Xinv","R","res")
    @printf("%11.5e %11.5e %11.5e %11.5e %11.5e %11.5e %11.5e %11.5e\n\n",time_total,timings[1:7]...)
    println("Time inside decomp:")
    @printf("%11s %11s %11s %11s %11s\n","schur","chol_S","comp CinvB","comp Q","chol_Q")
    @printf("%11.5e %11.5e %11.5e %11.5e %11.5e\n\n",timings[8:12]...)

    println("Time inside search directions (both predictor & corrector step)")
    @printf("%11s %11s %11s %11s %11s\n","calc Z","calc rhs x","solve system","calc dX","calc dY")
    @printf("%11.5e %11.5e %11.5e %11.5e %11.5e\n\n",timings[13:end]...)

    return x,X,y,Y,P,p,d, compute_duality_gap(constraints,x,y,Y,C,b),compute_primal_objective(constraints,x,b0), compute_dual_objective(y,Y,b,C,b0)
end

function compute_primal_objective(constraints,x,b0)
    return dot_c(constraints,x)+b0
end

function compute_dual_objective(y,Y,b,C,b0)
    return dot(C,Y)+dot(b,y)+b0 # block_diag_dot
end

function compute_primal_error(P,p)
    return max(abs.(p)...,[max(abs.(p)...) for p in P.blocks]...)
end

function compute_dual_error(d)
    return max(abs.(d)...)
end

function compute_duality_gap(constraints,x,y,Y,C,b)
    primal_objective = dot_c(constraints,x)
    dual_objective = dot(C,Y) + dot(b,y) # block_diag_dot, dot
    duality_gap = abs(primal_objective-dual_objective)/max(1,abs(primal_objective+dual_objective))
    return duality_gap
end

function dot_c(constraints,x)
    #compute dot(x,c) where c is distributed over the constraints
    c = vcat([constraints[j][3] for j=1:length(constraints)]...)
    return dot(c,x) #both ArbMatrices, so use the dot I defined
end

"""Compute the dual residue d = c- Tr(A_* Y) - By for this constraint."""
function calculate_res_d(constr, y, Y_block, blockinfo, j)
    # Calculate d = c - Tr(A_* Y) - By
    d = constr[3] - constr[2]*y # c - By
    return d - trace_A(constr,Y_block,blockinfo,j)
end


"""Compute the residuals P,p and d."""
function compute_residuals(constraints,x,X,y,Y,b,C,blockinfo)
    # P = ∑_i A_i x_i - X - C,
    P = -(X+C)
    add_weighted_A!(P,constraints,x,blockinfo)
    # println("res P get mid")
    Threads.@threads for j=1:blockinfo.J
        Arblib.get_mid!.(P.blocks[j].blocks,P.blocks[j].blocks)
    end

    # d = c- Tr(A_* Y) -By
    d = vcat([calculate_res_d(constraints[j],y,Y.blocks[j],blockinfo,j) for j=1:blockinfo.J]...) #vcat seems to work with Arbmatrices
    Arblib.get_mid!(d,d)
    # p = b -B^T x, B is distributed over constraints
    p = b
    p_added = [zero(p) for j=1:blockinfo.J] #zero also works with Arb

    # println("res p")
    Threads.@threads for j=1:blockinfo.J
        cur_x = 1
        j_idx = sum(blockinfo.dim_S[1:j-1])
        for r=1:blockinfo.m[j]
            for s=1:r
                for k=1:blockinfo.n_samples[j]
                    #add to p:
                    Arblib.addmul!(p_added[j],constraints[j][2][cur_x:cur_x,:],-x[cur_x+j_idx])
                    # p_added[j] -= x[cur_x+j_idx]*ArbMatrix(constraints[j][2][cur_x,:])
                    cur_x+=1 #x index for all r,s,k for this j
                end # of k
            end # of s
        end # of r
    end # end of j
    p +=sum(p_added)
    Arblib.get_mid!(p,p)

    return P,p,d
end

"""Determine whether the main loop should terminate or not"""
function terminate(duality_gap,primal_error,dual_error,duality_gap_threshold,primal_error_threshold,dual_error_threshold,need_primal_feasible,need_dual_feasible)
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
function check_pd_feasibility(primal_error,dual_error,primal_error_threshold,dual_error_threshold)
    primal_feas = primal_error < primal_error_threshold
    dual_feas = dual_error < dual_error_threshold
    return primal_feas && dual_feas
end


"""Compute the residual R, with or without second order term """
function compute_residual_R(X,Y,mu)
    # R = mu*I - XY
    R = similar(Y) #
    # println("R 1")
    Threads.@threads for j=1:length(Y.blocks)
        R.blocks[j] = mu*I + BlockDiagonal(.-(X.blocks[j].blocks .* Y.blocks[j].blocks))
        Arblib.get_mid!.(R.blocks[j].blocks,R.blocks[j].blocks)
    end

    return R
end

function compute_residual_R(X,Y,mu,dX,dY)
    # R = mu*I - XY -dXdY
    R = similar(Y)
    # println("R 2")
    Threads.@threads for j=1:length(Y.blocks)
        # we could use approx_mul! here, but that is a little more difficult. Would it matter?
        R.blocks[j] = mu*I + BlockDiagonal( .- X.blocks[j].blocks .* Y.blocks[j].blocks
                                            .- dX.blocks[j].blocks .* dY.blocks[j].blocks)
        Arblib.get_mid!.(R.blocks[j].blocks,R.blocks[j].blocks)
    end
    return R
end

"""Compute S, integrated with the precomputing of the bilinear pairings"""
function compute_S_integrated(constraints,X_inv,Y,blockinfo)
    S = [ArbMatrix(blockinfo.dim_S[j],blockinfo.dim_S[j],prec=precision(BigFloat)) for j=1:blockinfo.J]
    #bilinear pairings are only used per j,l. So we can compute them per j,l, and then make S_j
    # println("S int")
    Threads.@threads for j=1:blockinfo.J #or over the L? Not sure what is better
        for l=1:blockinfo.L[j]
            #compute the bilinear pairings:
            #NOTE: ranks,n_samples and m are constant per j,l. So we can in principle make a single big matrix with a little more complicated indexing
            #       Not sure if that helps with performance above this type of indexing (matrix in matrix in matrix)
            #initialize (doesnt cost a lot of time)
            vectors = hcat([constraints[j][1][l,k][rnk] for k=1:blockinfo.n_samples[j] for rnk=1:blockinfo.ranks[j][l]]...)
            delta = length(constraints[j][1][l,1][1])# all vectors in this block have this length
            vectors_trans = ArbMatrix(blockinfo.n_samples[j]*blockinfo.ranks[j][l],delta,prec = precision(BigFloat))
            Arblib.transpose!(vectors_trans,vectors)
            delta= length(constraints[j][1][l,1][1])# all vectors in this block have this length


            #might be able to parallelize making the matrix: first zeros/undef
            bilinear_pairings_Xinv = ArbMatrix(blockinfo.m[j]*blockinfo.n_samples[j]*blockinfo.ranks[j][l],blockinfo.m[j]*blockinfo.n_samples[j]*blockinfo.ranks[j][l],prec = precision(BigFloat))
            bilinear_pairings_Y = ArbMatrix(blockinfo.m[j]*blockinfo.n_samples[j]*blockinfo.ranks[j][l],blockinfo.m[j]*blockinfo.n_samples[j]*blockinfo.ranks[j][l],prec = precision(BigFloat))

            #no Arblib tensor/kronecker product, so we just do it per r,s block
            for s=1:blockinfo.m[j]
                part_matrix_Xinv = ArbMatrix(blockinfo.m[j]*delta,size(vectors,2),prec = precision(BigFloat))
                part_matrix_Y = ArbMatrix(blockinfo.m[j]*delta,size(vectors,2),prec=precision(BigFloat))
                Arblib.approx_mul!(part_matrix_Xinv,X_inv.blocks[j].blocks[l][:,(s-1)*delta+1:s*delta],vectors)
                Arblib.approx_mul!(part_matrix_Y,Y.blocks[j].blocks[l][:,(s-1)*delta+1:s*delta],vectors)

                # part_matrix_Xinv = X_inv.blocks[j].blocks[l][:,(s-1)*delta+1:s*delta] * vectors
                # part_matrix_Y = Y.blocks[j].blocks[l][:,(s-1)*delta+1:s*delta] * vectors

                for r=1:blockinfo.m[j] # we actually can do r=1:s, and then symmetrize?
                    Xinv_part = ArbMatrix(blockinfo.n_samples[j]*blockinfo.ranks[j][l],blockinfo.n_samples[j]*blockinfo.ranks[j][l],prec=precision(BigFloat))
                    Arblib.approx_mul!(Xinv_part,vectors_trans,part_matrix_Xinv[(r-1)*delta+1:r*delta,:])
                    bilinear_pairings_Xinv[(r-1)*blockinfo.n_samples[j]*blockinfo.ranks[j][l]+1:r*blockinfo.n_samples[j]*blockinfo.ranks[j][l],(s-1)*blockinfo.n_samples[j]*blockinfo.ranks[j][l]+1:s*blockinfo.n_samples[j]*blockinfo.ranks[j][l]] = Xinv_part

                    Y_part = ArbMatrix(blockinfo.n_samples[j]*blockinfo.ranks[j][l],blockinfo.n_samples[j]*blockinfo.ranks[j][l],prec=precision(BigFloat))
                    Arblib.approx_mul!(Y_part,vectors_trans,part_matrix_Y[(r-1)*delta+1:r*delta,:])
                    bilinear_pairings_Y[(r-1)*blockinfo.n_samples[j]*blockinfo.ranks[j][l]+1:r*blockinfo.n_samples[j]*blockinfo.ranks[j][l],(s-1)*blockinfo.n_samples[j]*blockinfo.ranks[j][l]+1:s*blockinfo.n_samples[j]*blockinfo.ranks[j][l]] = Y_part
                end
            end
            #clear the memory; probably unnecessary
            # Arblib.clear!(vectors_trans)
            # Arblib.clear!(vectors)
            # Arblib.clear!(part_matrix_Xinv)
            # Arblib.clear!(part_matrix_Y)

            #compute the contribution of this l to S[j]. Threads per j, so different threads dont write/read to/from the same matrices
            for r1 = 1:blockinfo.m[j]
                for s1=1:r1
                    for k1=1:blockinfo.n_samples[j]
                        #index for the tuple (r1,s1,k1):
                        hor_el = k1+((s1-1)+div(r1*(r1-1),2))*blockinfo.n_samples[j]
                        for r2=1:blockinfo.m[j]
                            for s2=1:r2
                                for k2 = 1:blockinfo.n_samples[j]
                                    #index for the tuple (r2,s2,k2):
                                    ver_el = k2+((s2-1)+div(r2*(r2-1),2))*blockinfo.n_samples[j]
                                    if ver_el <= hor_el #upper triangular part
                                        for rnk1=1:blockinfo.ranks[j][l],rnk2=1:blockinfo.ranks[j][l] #in one thread, we modify an element multiple times
                                            #calculate the entries of the bilinear pairing matrices corresponding to r1,s1,r2 and s2
                                            r1_spot = rnk1+blockinfo.ranks[j][l]*((k1-1)+ blockinfo.n_samples[j] *(r1-1))
                                            r2_spot = rnk2+blockinfo.ranks[j][l]*((k2-1)+ blockinfo.n_samples[j] *(r2-1))
                                            s1_spot = rnk1+blockinfo.ranks[j][l]*((k1-1)+ blockinfo.n_samples[j] *(s1-1))
                                            s2_spot = rnk2+blockinfo.ranks[j][l]*((k2-1)+ blockinfo.n_samples[j] *(s2-1))
                                            S[j][ver_el,hor_el] += constraints[j][4][l,k1][rnk1] * constraints[j][4][l,k2][rnk2]/Arb(4,prec=precision(BigFloat))*(
                                                bilinear_pairings_Xinv[s1_spot,r2_spot] * bilinear_pairings_Y[s2_spot,r1_spot]
                                                + bilinear_pairings_Xinv[r1_spot,r2_spot] * bilinear_pairings_Y[s2_spot,s1_spot]
                                                + bilinear_pairings_Xinv[s1_spot,s2_spot] * bilinear_pairings_Y[r2_spot,r1_spot]
                                                + bilinear_pairings_Xinv[r1_spot,s2_spot] * bilinear_pairings_Y[r2_spot,s1_spot] )
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            # Arblib.clear!(bilinear_pairings_Xinv)
            # Arblib.clear!(bilinear_pairings_Y)

        end
        S[j] .= Symmetric(S[j]) #symmetrize
        Arblib.get_mid!(S[j],S[j])
    end
    return S
end


"""Compute the decomposition of [S B; B^T 0]"""
function compute_T_decomposition(constraints,X_inv,Y,blockinfo)
    # 1) pre compute bilinear basis (uses constraints[j][1],X^{-1},Y)
    # 2) compute S (uses precomputed bilinear products)
    # 3) compute cholesky decomposition S =CC^T (uses S)
    # 4) compute decomposition of [S B; B^T 0] (uses constraints[j][2],S)

    # 1,2) compute the bilinear pairings and S, integrated. (per j,l, compute first pairings then S[j] part)
    time_schur = @elapsed begin
        S = compute_S_integrated(constraints,X_inv,Y,blockinfo)
    end

    # println("S,C")
    #3) cholesky decomposition of S
    time_cholS = @elapsed begin
        C = [similar(S[j]) for j=1:blockinfo.J]
        # println("S chol")
        Threads.@threads for j=1:blockinfo.J
            # Arblib.get_mid!(S[j],S[j])
            # we need cholesky, cannot use LU because we need CinvB.
            C[j] = ArbMatrix(cholesky(T.(S[j])).L.data) #do this cholesky with BigFloats
            # Arblib.get_mid!(C[j],C[j]) #ignore errors obtained from cholesky. Errors can get very large...
        end
    end

    #4) compute decomposition
    # CinvB has blocks C[j].L^-1 * B[j]
    time_CinvB = @elapsed begin
        CinvB = [ArbMatrix(blockinfo.dim_S[j],blockinfo.n_y,prec = precision(BigFloat)) for j=1:blockinfo.J]
        # println("CinvB")
        Threads.@threads for j=1:blockinfo.J
            Arblib.approx_solve_tril!(CinvB[j],C[j],constraints[j][2],0) #this threaded gives an error?
        end
    end
    # println("Q,Q_chol")
    #compute Q = B^T C.L^{-T} C.L^{-1}B
    #takes the most time (for multivariate).  May have advantage of reordering the constraints -> large constraints on separate threads.
    time_Q = @elapsed begin
        Q = [ArbMatrix(blockinfo.n_y,blockinfo.n_y,prec = precision(BigFloat)) for j=1:blockinfo.J]
        # println("comp Q")
        Threads.@threads for j=1:blockinfo.J
            #Note, if CinvB is ArbMatrix already, we still need to take ArbMatrix(transpose(CinvB[j])); matrix multiplication is not implemented for the adjoint in Arblib yet
            transp = ArbMatrix(size(CinvB[j],2),size(CinvB[j],1),prec = precision(BigFloat))
            Arblib.approx_mul!(Q[j],Arblib.transpose!(transp,CinvB[j]),CinvB[j])
        end
        Q = sum(Q)
        # Arblib.get_mid!(Q,Q)
    end

    # compute the cholesky factors of Q
    time_cholQ = @elapsed begin
        Q_chol = similar(Q)
        perm = [0 for i=1:size(Q,1)]
        Arblib.approx_lu!(perm,Q_chol,Q,prec = precision(Q))
        #Do approximate lu instead of cholesky. About as fast due to no error bounds, and more stable
        # Arblib.get_mid!(Q_chol,Q_chol) #ignore error bounds from cholesky
    end

    # C, CinvB, Q are the blocks that are used to build the decomposition.
    return (C,CinvB,perm,Q_chol), time_schur,time_cholS,time_CinvB,time_Q,time_cholQ
end

"""Compute the vector Tr(A_* Z) for one or all constraints"""
function trace_A(constraints,Z,blockinfo)
    #Assumption: Z is symmetric
    result = ArbMatrix(sum(blockinfo.dim_S),1,prec=precision(BigFloat))
    #one entry for each (j,r,s,k) tuple
    # println("trace A gen")
    Threads.@threads for j=1:blockinfo.J
        j_idx = sum(blockinfo.dim_S[1:j-1])
        tup_idx = 1
        for r = 1:blockinfo.m[j]
            for s=1:r
                for k=1:blockinfo.n_samples[j]
                    for l=1:blockinfo.L[j]
                        for rnk=1:blockinfo.ranks[j][l]
                            v = constraints[j][1][l,k][rnk] #we might want to store constraints things as arbmatrices...
                            delta = length(v)
                            v_transpose = ArbMatrix(1,delta,prec=precision(BigFloat))
                            Arblib.transpose!(v_transpose,v)
                            sgn = constraints[j][4][l,k][rnk]
                            # assumption:Z is symmetric. In that case Tr(vv^T Z[rs]) = Tr(vv^T Z[sr])
                            # Can we do this with larger matrix multiplications? Like the bilinear pairings?
                            # we can make a matrix V of k,rnk, and then do V^T Z^{j,l}_{r,s} V.
                            # But we need less, because we only need the diagonal of that.
                            # so we can do Z^{j,l}_{r,s} V and then take v^T * [column of Z corresponding to v]
                            # So basically precompute Z V per j,l,r,s and then add the things to the right element with a loop over k,rnk
                            result[j_idx+tup_idx,1] += sgn*(v_transpose*Z.blocks[j].blocks[l][(s-1)*delta+1:s*delta,(r-1)*delta+1:r*delta]*v)[1,1]
                        end
                    end
                    tup_idx += 1 #tuple (r,s,k)
                end
            end
        end
    end
    return result
end

function trace_A(constraint,Z,blockinfo,j)
    #Assumption: Z is symmetric. Tr(Z λvv^T 1/2(e_r e_s + e_s e_r)) =1/2 λ(Tr(Z[r,s]vv^T +Z[s,r]vv^T)) = λv^TZ[r,s]v
    result = ArbMatrix(sum(blockinfo.dim_S[j]),1,prec = precision(BigFloat))
    tup_idx = 1
    #we can do this threaded (over n_samples for example) by calculating tup_idx instead of doing +1 every iteration
    for r = 1:blockinfo.m[j]
        for s=1:r
            for k=1:blockinfo.n_samples[j]
                for l=1:blockinfo.L[j]
                    for rnk=1:blockinfo.ranks[j][l]
                        v = constraint[1][l,k][rnk]
                        delta = length(v)
                        v_transpose = ArbMatrix(1,delta,prec=precision(BigFloat))
                        Arblib.transpose!(v_transpose,v)
                        sgn = constraint[4][l,k][rnk]
                        result[tup_idx,1] += sgn*(v_transpose*Z.blocks[l][(r-1)*delta+1:r*delta,(s-1)*delta+1:s*delta]*v)[1,1]
                    end
                end
                tup_idx +=1
            end
        end
    end
    return result
end

"""Add ∑_i a_i A_i to initial_matrix"""
function add_weighted_A!(initial_matrix,constraints,a,blockinfo)
    #initial matrix is block matrix of ArbMatrices
    #NOTE: instead of adding Q to both the r,s block and the s,r block, we can add it to the upper block and use Symmetric()
    # println("add weighted A")
    Threads.@threads for j=1:blockinfo.J
        j_idx = sum(blockinfo.dim_S[1:j-1])
        cur_a = 1
        for l = 1:blockinfo.L[j]
            delta = length(constraints[j][1][l,1][1])
            for r=1:blockinfo.m[j]
                for s=1:r
                    Q = ArbMatrix(delta,delta,prec= precision(BigFloat))
                    for k=1:blockinfo.n_samples[j]
                        cur_a = k+(s-1+div(r*(r-1),2))*blockinfo.n_samples[j]
                        for rnk=1:blockinfo.ranks[j][l]
                            v = constraints[j][1][l,k][rnk] #v_{j,l,k,i}
                            sgn = constraints[j][4][l,k][rnk]*a[cur_a+j_idx]/Arb(2,prec=precision(BigFloat))
                            #the Q which should be added to the j,l,(r,s) and (s,r) block of P
                            #a[i]*v*v' is not always symmetric, a[i]*(v*v') is.
                            v_transpose = ArbMatrix(1,delta,prec=precision(BigFloat))
                            Arblib.transpose!(v_transpose,v)
                            Arblib.addmul!(Q,v*v_transpose,sgn)
                            # Q += Symmetric(sgn/2*a[cur_a+j_idx]*(v*v_transpose)) #expect a[i] to be Arb
                        end
                        cur_a+=1 # tuple (r,s,k)
                    end
                    if r != s
                        # initial_matrix.blocks[j].blocks[l][(r-1)*delta+1:r*delta, (s-1)*delta+1:s*delta] += Q
                        #Arblib.addmul!(B,A,c) => B = B+ c*A
                        initial_matrix.blocks[j].blocks[l][(s-1)*delta+1:s*delta, (r-1)*delta+1:r*delta] += Q #Q = Q'# works
                    else
                        Arblib.mul!(Q,Q,2) #Q -> 2Q, which has to be added to this block
                        initial_matrix.blocks[j].blocks[l][(r-1)*delta+1:r*delta, (r-1)*delta+1:r*delta] += Q #Q = Q' Maybe addmul is more efficient?
                    end
                end
            end
            initial_matrix.blocks[j].blocks[l] .= Symmetric(initial_matrix.blocks[j].blocks[l])
        end
    end
    return nothing #initial matrix modified, nothing returned
end

"""Compute the search directions, using a precomputed decomposition"""
function compute_search_direction(constraints,P,p,d,R,X_inv,Y,blockinfo,(C,CinvB,perm,Q))
    # using the decomposition, compute the search directions
    # 5) solve system with rhs dx,dy <- (-d - Tr(A_* Z) ;  p) with Z = X^{-1}(PY - R)
    #NOTE:  computing rhs_x (including Z) only needs to be done once; so we may want to do that before the function
    # 6) compute dX = P + sum_i A_i dx_i
    # 7) compute dY = X^{-1}(R-dX Y) (XdY = R-dXY)
    # 8) symmetrize dY = 1/2 (dY +dY')
    time_Z = @elapsed begin
        Z = similar(Y)
        # println("Z")
        Threads.@threads for j=1:blockinfo.J
            #Z = X_inv*(P*Y-R)
            for l=1:blockinfo.L[j]
                # temp = similar(Z.blocks[j].blocks[l])
                Arblib.approx_mul!(Z.blocks[j].blocks[l],P.blocks[j].blocks[l],Y.blocks[j].blocks[l])
                Arblib.sub!(Z.blocks[j].blocks[l],Z.blocks[j].blocks[l],R.blocks[j].blocks[l])
                Arblib.approx_mul!(Z.blocks[j].blocks[l],X_inv.blocks[j].blocks[l],Z.blocks[j].blocks[l])
                temp_t = similar(Z.blocks[j].blocks[l])
                Arblib.transpose!(temp_t,Z.blocks[j].blocks[l])
                Arblib.mul!(Z.blocks[j].blocks[l],Z.blocks[j].blocks[l]+temp_t,Arb(1//2,prec=precision(BigFloat)))
                Arblib.get_mid!(Z.blocks[j].blocks[l],Z.blocks[j].blocks[l])
            end
        end
    end
    rhs_y = p
    time_rhs_x = @elapsed begin
        rhs_x = similar(d)
        rhs_x = -d - trace_A(constraints,Z,blockinfo)
        Arblib.get_mid!(rhs_x,rhs_x)
    end

    # solve the system (C 0; CinvB^T I)(I 0; 0 LL^T)(C^T -CinvB; 0 I)(dx; dy) = (rhs_x; rhs_y)
    indices = blockinfo.x_indices #0, dim_S[1], dim_S[1]+dim_S[2],... ,sum(dim_S)
    time_sys = @elapsed begin
        #first lower triangular system:
        temp_x = [ArbMatrix(indices[j+1]-indices[j],1,prec=precision(BigFloat)) for j=1:blockinfo.J]
        temp_y = [similar(rhs_y) for j=1:blockinfo.J]
        # println("solve sys")
        Threads.@threads for j=1:blockinfo.J
            Arblib.approx_solve_tril!(temp_x[j],C[j],rhs_x[indices[j]+1:indices[j+1],:],0)
            transp = ArbMatrix(size(CinvB[j],2),size(CinvB[j],1),prec=precision(BigFloat))
            Arblib.approx_mul!(temp_y[j], Arblib.transpose!(transp,CinvB[j]),temp_x[j]) #now its an ArbMatrix, otherwise ArbVector which uses generic multipilcations
        end
        dy = similar(rhs_y)
        Arblib.sub!(dy,rhs_y,sum(temp_y))
        # dy = rhs_y - sum(temp_y)
        #second system: temp_x stays the same, dy_new  =  Q^-1 dy_old
        Arblib.approx_solve_lu_precomp!(dy,perm,Q,dy)

        #third system:
        dx = [ArbMatrix(indices[j+1]-indices[j],1,prec=precision(BigFloat)) for j=1:blockinfo.J]
        # println("solve sys 2")
        Threads.@threads for j=1:blockinfo.J
            temp_C = similar(C[j])
            # Arblib.addmul!(temp_x[j],CinvB[j],dy)
            Arblib.approx_solve_triu!(dx[j],Arblib.transpose!(temp_C,C[j]),(temp_x[j] + CinvB[j] * dy),0)
        end
        dx = vcat([dx[j] for j=1:blockinfo.J]...)
        Arblib.get_mid!(dx,dx) #not sure if this is needed, because we used approx_solve_triu! (i.e. the approx version)
    end #of timing system

    #step 6:
    time_dX = @elapsed begin
        dX = similar(P) # making sure dX does not reference to P anymore
        dX += P
        add_weighted_A!(dX,constraints, dx,blockinfo)
    end

    #step 7 & 8
    time_dY = @elapsed begin
        dY = similar(Y)
        # println("dY")
        Threads.@threads for j=1:blockinfo.J
            #dY = X_inv * (R- dX *Y)
            for l=1:blockinfo.L[j]
                # temp = similar(dY.blocks[j].blocks[l])
                Arblib.approx_mul!(dY.blocks[j].blocks[l],dX.blocks[j].blocks[l],Y.blocks[j].blocks[l])
                Arblib.sub!(dY.blocks[j].blocks[l],R.blocks[j].blocks[l],dY.blocks[j].blocks[l])
                Arblib.approx_mul!(dY.blocks[j].blocks[l],X_inv.blocks[j].blocks[l],dY.blocks[j].blocks[l])
                temp_t = similar(dY.blocks[j].blocks[l])
                Arblib.transpose!(temp_t,dY.blocks[j].blocks[l])
                Arblib.mul!(dY.blocks[j].blocks[l],dY.blocks[j].blocks[l]+temp_t,Arb(1//2,prec=precision(BigFloat)))
                Arblib.get_mid!(dY.blocks[j].blocks[l],dY.blocks[j].blocks[l])
                Arblib.get_mid!(dX.blocks[j].blocks[l],dX.blocks[j].blocks[l])
            end

        end
    end

    return dx,dX,dy,dY, [time_Z,time_rhs_x,time_sys,time_dX,time_dY]
end


"""Compute the step length min(γ α(M,dM), 1), where α is the maximum number step
to which keeps M+α(M,dM) dM positive semidefinite"""
function compute_step_length(M::BlockDiagonal,dM::BlockDiagonal,gamma)
    # # cholb = [[cholesky(T.(m.blocks[l])) for l=1:length(m.blocks)] for m in blocks(M)]
    min_eig = [T(Inf) for j=1:length(M.blocks)]

    # min_eig = [Arb(Inf) for j=1:length(M.blocks)]
    # println("ev")
    Threads.@threads for j in 1:length(M.blocks)
        for l in 1:length(M.blocks[j].blocks)
            # chol = similar(M.blocks[j].blocks[l])
            # Arblib.cho!(chol,M.blocks[j].blocks[l])
            # LML = similar(dM.blocks[j].blocks[l])
            # Arblib.approx_solve_tril!(LML, chol, dM.blocks[j].blocks[l],0) #LML = chol^-1 dMblock
            # Arblib.transpose!(LML,LML)
            # tempLML = similar(LML)
            # Arblib.approx_solve_tril!(tempLML, chol, LML,0) #temp LML = chol^-1 (chol^-1 dMblock)^T = chol^-1
            # eigenvalues = AcbVector(size(LML,1))
            # Arblib.approx_eig_qr!(eigenvalues,AcbMatrix(tempLML))
            # min_eig[j] = min(real.(eigenvalues)..., min_eig[j])
            cholb = cholesky(T.(M.blocks[j].blocks[l]))
            LMLb = cholb.L \ T.(dM.blocks[j].blocks[l]) / cholb.U
            min_eig[j] = min(eigmin((LMLb+LMLb')/2), min_eig[j])
        end
    end
    min_eig = min(min_eig...)

    if min_eig > -gamma
        return Arb(1,prec=precision(BigFloat))
    else
        return Arb(-gamma/min_eig,prec=precision(BigFloat))
    end
end


end #of module MPMP
