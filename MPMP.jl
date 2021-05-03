module MPMP

using AbstractAlgebra, Combinatorics
using GenericLinearAlgebra, LinearAlgebra
using BlockDiagonals
using Printf
using GenericSVD #SVD of bigfloat matrix, for prepareabc with rank>1 structure
using Arblib #lightweight implementation of Arb, in development

#need GenericLinearAlgebra for BigFloat support (eigmin)
import LinearAlgebra.dot
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

function points_X_general(n,d)# sometimes work, not always.
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
    #roots of chebyshev polynomials of the first kind
    return [(a+b)/BigFloat(2) + (b-a)/BigFloat(2) * cos((2k-1)/BigFloat(2(d+1))*BigFloat(pi)) for k=1:d+1]
end

function create_sample_points_chebyshev_mod(d,a=-1,b=1)
    #roots of chebyshev polynomials of the first kind, divided by cos(pi/2(d+1)) to get a lower lebesgue constant
    return [(a+b)/BigFloat(2) + (b-a)/BigFloat(2) * cos((2k-1)/BigFloat(2(d+1))*BigFloat(pi))/cos(BigFloat(pi)/2(d+1)) for k=1:d+1]
end

## Functions for the solver

#extending LinearAlgebra.dot for our BlockDiagonal matrices. in principle 'type piracy'
function dot(A::BlockDiagonal, B::BlockDiagonal)
   #assume that A and B have the same blockstructure
   sum(dot(a,b) for (a,b) in zip(blocks(A), blocks(B)))
end
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
                    Pi = nothing;  #polynomial matrices, as much as G has. We will use A_(jrsk) = sum_l Tr(Q ⊗ E_rs) where Q = ∑_η λ_η(x_k)(√G[l](x_k)q(x_k) ⊗ v_η(x_k)) (√G[l](x_k)q(x_k) ⊗ v_η(x_k))^T
                    normalize = false) # normalize constraint such that the highest
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
        Pi_vecs = [ [svd_decomps[l,k].U[r,:] for r=1:size(Pi[l],1)] for l=1:length(G), k=1:length(x)]
        Pi_vals = [ [dot(svd_decomps[l,k].U[r,:],svd_decomps[l,k].Vt[r,:])*svd_decomps[l,k].S[r] for r=1:size(Pi[l],1)] for l=1:length(G), k=1:length(x)]
        deg_Pi = [max([total_degree(Pi[l][i,j]) for i=1:size(Pi[l],1) for j=1:size(Pi[l],2)]...) for l=1:length(G)]
    end

    # find the first occurance of a degree in the basis. Needed for symmetries, where the number of required basis polynomials can be (much) smaller than (n+d choose d)
    #
    degrees = ones(Int64,div(δ,2)+1) #maximum degree needed is δ/2. everything is an index, so at least 1
    cur_deg = 0
    for d = 1:length(q)
        if total_degree(q[d])==cur_deg
            degrees[cur_deg+1] = d
            cur_deg+=1
            if cur_deg+1 > length(degrees) #only need degrees up to div(δ,2)
                break
            end
        elseif total_degree(q[d])<cur_deg-1 || total_degree(q[d])>cur_deg #degree can be cur_degree-1 (still basis for prev part) or cur_degree (start of basis for new part)
            println("Something might be wrong, the degree of q is not monotone increasing with steps of 1")
        end
    end
    # println(degrees)
    # We can even put the whole G[l](x[k]...) in the 'sign'. We already have the eigenvalues of the Pi there
    # either way, better to be consistent (either also G in A_sign, or only the signs -> ev of Pi in A)
    A_sign = [[T(Pi_vals[l,k][r]*sign(G[l](x[k]...))) for r=1:length(Pi_vals[l,k])] for l=1:length(G),k=1:length(x)]
    A = [ [T.(kron([q[d](x[k]...)*sqrt(abs(G[l](x[k]...))) for d=1:degrees[div(δ-total_degree(G[l])-deg_Pi[l],2)+1]],Pi_vecs[l,k][r])) for r=1:length(Pi_vecs[l,k])] for l=1:length(G), k=1:length(x)]
    # A[l,k][r] is the vector v_{j,l,k,r} for this constraint j
    # A_sign[l,k][r] gives the sign and the r'th eigenvalue of Pi. i.e Q = \sum_r A_sign[l,k][r] A[l,k][r] *A[l,k][r]'

    B = vcat([transpose([T(-M[i][r,s](x[k]...)) for i=2:length(M)]) for r=1:m for s=1:r for k=1:length(x)]...)

    c = vcat([[T(M[1][r,s](x[k]...))] for r=1:m for s=1:r for k=1:length(x)]...)
    if normalize
        for k=1:length(x)
            max_abs = BigFloat(0)
            for l=1:length(G)
                for r=1:length(Pi_vals[l,k])
                    max_vec = max(abs.(A_sign[l,k][r]*(A[l,k][r]*A[l,k][r]'))...)
                    max_abs = max(max_abs,max_vec)
                end
            end
            max_abs = max(max_abs,abs.(B[k:div(m*(m+1),2):end,:])...,abs.(c[k:div(m*(m+1),2):end])...)*100.0

            for l=1:length(G)
                for r=1:length(Pi_vals[l,k])
                    A_sign[l,k][r] /= max_abs
                    A[l,k][r] ./= max_abs
                end
            end
            B[k:div(m*(m+1),2):end,:] ./= max_abs
            c[k:div(m*(m+1),2):end] ./= max_abs
        end
    end
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
                   b,Pi=nothing;normalize_prep = false, kwargs...) # Objective vector
    if !isnothing(Pi)
        abc = [prepareabc(M[j],G[j],q[j],x[j],delta[j],Pi[j],normalize = normalize_prep) for j=1:length(M)]
    else
        abc = [prepareabc(M[j],G[j],q[j],x[j],delta[j],normalize = normalize_prep) for j=1:length(M)]
    end
    # Use prepareabc to call solverank1sdp.
    blockinfo = get_block_info(abc)
    solverank1sdp(abc, b, blockinfo; kwargs...)
end

#to use C = 0 efficiently. Does not really matter for performance
struct AbsoluteZero end
LinearAlgebra.dot(x::AbsoluteZero,y) = eltype(y)(0)
Base.:+(X::BlockDiagonal, C::AbsoluteZero) = X



"""Solve the SDP with rank one constraint matrices."""
function solverank1sdp(constraints, # list of (A,B,c,H) tuples (eltype = T)
                       b, # Objective vector
                       blockinfo; # information about the block sizes etc.
                       C=0, b0 = 0, maxiterations=500,
                       beta_infeas = T(3)/10,
                       beta_feas= T(1)/10,
                       gamma =  T(7)/10,
                       omega_p = T(10)^(10), #in general, can be chosen smaller. might need to be increased in some cases
                       omega_d = T(10)^(10),
                       duality_gap_threshold = T(10)^(-30),
                       primal_error_threshold = T(10)^(-30),
                       dual_error_threshold = T(10)^(-30),
                       need_primal_feasible = false,
                       need_dual_feasible = false,
                       testing = false)
                       #the defaultvalues mostly from Simmons-Duffin original paper
    #convert to BigFloats:
    b = T.(b)
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
    x = zeros(T,sum(blockinfo.dim_S)) # all tuples (j,r,s,k), equals size of S.
    X = T(omega_p)*BlockDiagonal([BlockDiagonal([Matrix{T}(I,blockinfo.Y_blocksizes[j][l],blockinfo.Y_blocksizes[j][l]) for l=1:blockinfo.L[j]]) for j=1:blockinfo.J])
    y = zeros(T,blockinfo.n_y)
    Y = T(omega_d)*BlockDiagonal([BlockDiagonal([Matrix{T}(I,blockinfo.Y_blocksizes[j][l],blockinfo.Y_blocksizes[j][l]) for l=1:blockinfo.L[j]]) for j=1:blockinfo.J])
    if C == 0 #no C objective given. #in principle we can remove C in most cases. Only for computing residuals; not sure on the impact regarding memory (as it is always zero)
        # C = zero(Y) #make sure adding/subtracting C works in case it is 0.
        C = AbsoluteZero() # works
        #Can we make it a 'special' zero which does nothing and requires no memory? So define X + C = X for all X, and dot(C,Y) = 0 for all Y
    end


    #step 2
    time_res = @elapsed begin
        P,p,d = compute_residuals(constraints,x,X,y,Y,b,C,blockinfo)
        pd_feas = check_pd_feasibility(P,p,d,primal_error_threshold,dual_error_threshold)
    end

    #loop initialization
    iter = 1
    @printf("%5s %8s %11s %11s %11s %10s %10s %10s %10s %10s %10s %10s\n", "iter","time(s)","μ", "P-obj","D-obj","gap","P-error","p-error","d-error","α_p","α_d","beta")
    alpha_p = alpha_d = T(0)
    mu = dot(X,Y)/size(X,1) #block_diag_dot
    p_obj = compute_primal_objective(constraints,x,b0)
    d_obj = compute_dual_objective(y,Y,b,C,b0)
    dual_gap = compute_duality_gap(constraints,x,y,Y,C,b)
    timings = zeros(Float64,17) #timings do not require high precision
    time_start = time()

    while (!terminate(constraints,P,p,d,x,y,Y,C,b,duality_gap_threshold,primal_error_threshold,dual_error_threshold,need_primal_feasible,need_dual_feasible)
        && iter < maxiterations)
        #step 3
        mu = dot(X,Y)/size(X,1) # block_diag_dot
        mu_p =  pd_feas ? zero(mu) : beta_infeas * mu

        #step 4
        time_R = @elapsed begin
            R = compute_residual_R(X,Y,mu_p)
        end

        time_inv = @elapsed begin
            X_inv = BlockDiagonal([BlockDiagonal(inv.(cholesky.(blocks(bx)))) for bx in blocks(X)])
        end
        time_decomp = @elapsed begin
            decomposition, time_schur,time_cholS,time_CinvB, time_Q,time_cholQ = compute_T_decomposition(constraints,X_inv,Y,blockinfo)
        end

        time_predictor_dir = @elapsed begin
            dx,dX,dy,dY,times_predictor_in = compute_search_direction(constraints,P,p,d,R,X_inv,Y,blockinfo,decomposition)
        end

        #step 5
        r = dot(X+dX,Y+dY)/(mu*size(X,1)) # block_diag_dot
        beta = r<1 ? r^2 : r
        beta_c = pd_feas ? min(max(beta_feas,beta),T(1)) : max(beta_infeas, beta)
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
        #if pd feasible, follow search direction exactly. (follows Simmons duffins code)
        if pd_feas
            alpha_p = min(alpha_p,alpha_d)
            alpha_d = alpha_p
        end
        #step 8
        x += alpha_p*dx
        X += alpha_p*dX
        y += alpha_d*dy
        Y += alpha_d*dY
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

        #step 2, preparation for new loop iteration
        iter+=1
        time_res = @elapsed begin
            P,p,d = compute_residuals(constraints, x,X,y,Y,b,C,blockinfo)
        end
        pd_feas = check_pd_feasibility(P,p,d,primal_error_threshold,dual_error_threshold)
    end
    time_total = time()-time_start
    @printf("%5s %8s %11s %11s %11s %10s %10s %10s %10s %10s %10s %10s\n", "iter","time(s)","μ", "P-obj","D-obj","gap","P-error","p-error","d-error","α_p","α_d","beta")

    #print the total time needed for every part of the algorithm
    println("\nTime spent: ")
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
    x_idx=1
    res = eltype(x)(0)
    for j=1:length(constraints)
        res+=dot(constraints[j][3],x[x_idx:x_idx+length(constraints[j][3])-1])
        x_idx+=length(constraints[j][3])
    end
    return res
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

    # d = c- Tr(A_* Y) -By
    d = vcat([calculate_res_d(constraints[j],y,Y.blocks[j],blockinfo,j) for j=1:blockinfo.J]...)

    # p = b -B^T x, B is distributed over constraints
    p = b
    p_added = [zero(p) for j=1:blockinfo.J] #for parallelization
    Threads.@threads for j=1:blockinfo.J
        cur_x = 1
        j_idx = sum(blockinfo.dim_S[1:j-1])
        for r=1:blockinfo.m[j]
            for s=1:r
                for k=1:blockinfo.n_samples[j]
                    #add to p:
                    p_added[j] -= x[cur_x+j_idx]*constraints[j][2][cur_x,:]
                    cur_x+=1 #x index for all r,s,k for this j
                end # of k
            end # of s
        end # of r
    end # end of j
    p +=sum(p_added)

    return P,p,d
end

"""Determine whether the main loop should terminate or not"""
function terminate(constraints,P,p,d,x,y,Y,C,b,duality_gap_threshold,primal_error_threshold,dual_error_threshold,need_primal_feasible,need_dual_feasible)
    duality_gap_opt = compute_duality_gap(constraints,x,y,Y,C,b) < duality_gap_threshold
    primal_feas = compute_primal_error(P,p) < primal_error_threshold
    dual_feas = compute_dual_error(d) < dual_error_threshold
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
function check_pd_feasibility(P,p,d,primal_error_threshold,dual_error_threshold)
    primal_feas = compute_primal_error(P,p) < primal_error_threshold
    dual_feas = compute_dual_error(d) < dual_error_threshold
    return primal_feas && dual_feas
end


"""Compute the residual R, with or without second order term """
function compute_residual_R(X,Y,mu)
    # R = mu*I - XY
    R = similar(Y)
    Threads.@threads for j=1:length(X.blocks)
        R.blocks[j] = mu*I + BlockDiagonal(.-(X.blocks[j].blocks .* Y.blocks[j].blocks))
    end
    return R
end

function compute_residual_R(X,Y,mu,dX,dY)
    # R = mu*I - XY -dXdY
    R = similar(Y)
    Threads.@threads for j=1:length(X.blocks)
        R.blocks[j] = mu*I + BlockDiagonal( .- X.blocks[j].blocks .* Y.blocks[j].blocks
                                            .- dX.blocks[j].blocks .* dY.blocks[j].blocks)
    end
    return R
end

"""Compute S, integrated with the precomputing of the bilinear pairings"""
function compute_S_integrated(constraints,X_inv,Y,blockinfo)
    S = [zeros(T, blockinfo.dim_S[j],blockinfo.dim_S[j] ) for j=1:blockinfo.J]
    #bilinear pairings are only used per j,l. So we can compute them per j,l, and then make S_j
    for j=1:blockinfo.J
        for l=1:blockinfo.L[j]
            #compute the bilinear pairings:
            #NOTE: ranks,n_samples and m are constant per j,l. So we can in principle make a single big matrix with a little more complicated indexing
            #       Not sure if that helps with performance above this type of indexing (matrix in matrix in matrix)
            bilinear_Y = [[zeros(T,blockinfo.m[j],blockinfo.m[j]) for k1=1:blockinfo.n_samples[j], k2=1:blockinfo.n_samples[j]] for rnk1=1:blockinfo.ranks[j][l],rnk2=1:blockinfo.ranks[j][l]]
            bilinear_Xinv = [[zeros(T,blockinfo.m[j],blockinfo.m[j]) for k1=1:blockinfo.n_samples[j], k2=1:blockinfo.n_samples[j]] for rnk1=1:blockinfo.ranks[j][l],rnk2=1:blockinfo.ranks[j][l]]
            Threads.@threads for k1=1:blockinfo.n_samples[j]
                for rnk1 = 1:blockinfo.ranks[j][l]
                    #we can do Xv already here, then use the relevant part of that after s,k2
                    # in principle that wont matter for theoretical complexity.
                    v1 = constraints[j][1][l,k1][rnk1]
                    delta = length(v1)
                    for r=1:blockinfo.m[j]
                        for s = 1:r
                            Xv = X_inv.blocks[j].blocks[l][(r-1)*delta+1:r*delta,(s-1)*delta+1:s*delta] * v1
                            Yv = Y.blocks[j].blocks[l][(r-1)*delta+1:r*delta,(s-1)*delta+1:s*delta] * v1
                            if s!= r
                                #only needed for off diagonal blocks (originally v1' * the same block. for dot(v1,block(r,s),v2) = dot(v2,block(s,r),v1))
                                vX = X_inv.blocks[j].blocks[l][(s-1)*delta+1:s*delta,(r-1)*delta+1:r*delta]* v1
                                vY = Y.blocks[j].blocks[l][(s-1)*delta+1:s*delta,(r-1)*delta+1:r*delta] * v1
                            end
                            for k2=1:k1 #do k1,k2 and k2,k1 at the same time
                                for rnk2=1:blockinfo.ranks[j][l]
                                    v2 = constraints[j][1][l,k2][rnk2]
                                    bilinear_Xinv[rnk1,rnk2][k1,k2][s,r] = dot(v2,Xv) #equivalent to dot(v1,Xinv_block',v2)
                                    if s!=r
                                        bilinear_Xinv[rnk1,rnk2][k1,k2][r,s] = dot(vX,v2)
                                    end
                                    if k1!= k2
                                        bilinear_Xinv[rnk2,rnk1][k2,k1][r,s] = bilinear_Xinv[rnk1,rnk2][k1,k2][s,r]
                                        if s!= r
                                            bilinear_Xinv[rnk2,rnk1][k2,k1][s,r] = bilinear_Xinv[rnk1,rnk2][k1,k2][r,s]
                                        end
                                    end
                                    bilinear_Y[rnk1,rnk2][k1,k2][s,r] = dot(v2,Yv)
                                    if s!=r
                                        bilinear_Y[rnk1,rnk2][k1,k2][r,s] = dot(vY,v2)
                                    end
                                    if k1!= k2
                                        bilinear_Y[rnk2,rnk1][k2,k1][r,s] = bilinear_Y[rnk1,rnk2][k1,k2][s,r]
                                        if s!= r
                                            bilinear_Y[rnk2,rnk1][k2,k1][s,r] = bilinear_Y[rnk1,rnk2][k1,k2][r,s]
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end

            #compute the contribution of this l to S[j]
            for r1 = 1:blockinfo.m[j]
                for s1=1:r1
                    Threads.@threads for k1=1:blockinfo.n_samples[j]
                        #index for the tuple (r1,s1,k1):
                        hor_el = k1+((s1-1)+div(r1*(r1-1),2))*blockinfo.n_samples[j]
                        for r2=1:blockinfo.m[j]
                            for s2=1:r2
                                for k2 = 1:blockinfo.n_samples[j]
                                    #index for the tuple (r2,s2,k2):
                                    ver_el = k2+((s2-1)+div(r2*(r2-1),2))*blockinfo.n_samples[j]
                                    if ver_el <= hor_el #upper triangular part
                                        for rnk1=1:blockinfo.ranks[j][l],rnk2=1:blockinfo.ranks[j][l] #in one thread, we modify an element multiple times
                                            # sgn = constraints[j][4][l,k1][rnk1] * constraints[j][4][l,k2][rnk2]
                                            #indexing is[rnk1,rnk2][k1,k2][r,s]. rnk1 corresponds to k1, rnk2 to k2
                                            S[j][ver_el,hor_el] += constraints[j][4][l,k1][rnk1] * constraints[j][4][l,k2][rnk2]/T(4)*(
                                                bilinear_Xinv[rnk1,rnk2][k1,k2][s1,r2] * bilinear_Y[rnk2,rnk1][k2,k1][s2,r1]
                                                + bilinear_Xinv[rnk1,rnk2][k1,k2][r1,r2] * bilinear_Y[rnk2,rnk1][k2,k1][s2,s1]
                                                + bilinear_Xinv[rnk1,rnk2][k1,k2][s1,s2] * bilinear_Y[rnk2,rnk1][k2,k1][r2,r1]
                                                + bilinear_Xinv[rnk1,rnk2][k1,k2][r1,s2] * bilinear_Y[rnk2,rnk1][k2,k1][r2,s1] )
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            bilinear_Y = nothing
            bilinear_Xinv = nothing
        end
    end
    return Symmetric.(S) # the blocks S[j], symmetrized
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

    #3) cholesky decomposition of S
    time_cholS = @elapsed begin
        C1 = cholesky(S[1])
        C = [C1 for j=1:blockinfo.J] #initialize.
        Threads.@threads for j=2:blockinfo.J
            C[j] = cholesky(S[j])
        end
    end

    #4) compute decomposition
    # CinvB has blocks C[j].L^-1 * B[j]
    time_CinvB = @elapsed begin
        CinvB = [zeros(T,blockinfo.dim_S[j],blockinfo.n_y) for j=1:blockinfo.J]
        Threads.@threads for j=1:blockinfo.J
            CinvB[j] = C[j].L \ constraints[j][2]
        end
    end

    #compute Q = B^T C.L^{-T} C.L^{-1}B
    #takes the most time (for multivariate).  May have advantage of reordering the constraints -> large constraints on separate threads.
    time_Q = @elapsed begin
        Q = [ArbMatrix(zeros(T,blockinfo.n_y,blockinfo.n_y),prec = precision(BigFloat)) for j=1:blockinfo.J]
        Threads.@threads for j=1:blockinfo.J
            #Note, if CinvB is ArbMatrix already, we still need to take ArbMatrix(transpose(CinvB[j])); matrix multiplication is not implemented for the adjoint in Arblib yet
            Q[j] = ArbMatrix(transpose(CinvB[j]),prec=precision(BigFloat)) * ArbMatrix(CinvB[j],prec=precision(BigFloat))
        end
        #go back to bigfloats for cholesky etc
        Q = BigFloat.(sum(Q))
    end

    # compute the cholesky factors of Q
    time_cholQ = @elapsed begin
        Q_chol = cholesky(Q)
    end

    # C, CinvB, Q are the blocks that are used to build the decomposition.
    return (C,CinvB,Q_chol), time_schur,time_cholS,time_CinvB,time_Q,time_cholQ
end

"""Compute the vector Tr(A_* Z) for one or all constraints"""
function trace_A(constraints,Z,blockinfo)
    #Assumption: Z is symmetric
    result = zeros(T,sum(blockinfo.dim_S)) #one entry for each (j,r,s,k) tuple
    Threads.@threads for j in eachindex(constraints)
        j_idx = sum(blockinfo.dim_S[1:j-1])
        tup_idx = 1
        for r = 1:blockinfo.m[j]
            for s=1:r
                for k=1:blockinfo.n_samples[j]
                    for l=1:blockinfo.L[j]
                        for rnk=1:blockinfo.ranks[j][l]
                            v = constraints[j][1][l,k][rnk]
                            sgn = constraints[j][4][l,k][rnk]
                            delta = length(v)
                            # assumption:Z is symmetric. In that case Tr(vv^T Z[rs]) = Tr(vv^T Z[sr])
                            result[j_idx+tup_idx] += sgn*dot(v,Z.blocks[j].blocks[l][(s-1)*delta+1:s*delta,(r-1)*delta+1:r*delta],v)
                        end
                    end
                    tup_idx += 1
                end
            end
        end
    end
    return result
end

function trace_A(constraint,Z,blockinfo,j)
    #Assumption: Z is symmetric
    result = zeros(T,sum(blockinfo.dim_S[j]))
    tup_idx = 1
    for r = 1:blockinfo.m[j]
        for s=1:r
            for k=1:blockinfo.n_samples[j]
                for l=1:blockinfo.L[j]
                    for rnk=1:blockinfo.ranks[j][l]
                        v = constraint[1][l,k][rnk]
                        sgn = constraint[4][l,k][rnk]
                        delta = length(v)
                        result[tup_idx]+= sgn*dot(v,Z.blocks[l][(r-1)*delta+1:r*delta,(s-1)*delta+1:s*delta],v)
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
    #NOTE: instead of adding Q to both the r,s block and the s,r block, we can add it to the upper block and use Symmetric()
    # Saves some time, but not in place anymore (afaik)
    Threads.@threads for j=1:blockinfo.J
        j_idx = sum(blockinfo.dim_S[1:j-1])
        cur_a = 1
        for l = 1:blockinfo.L[j]
            for r=1:blockinfo.m[j]
                for s=1:r
                    delta = length(constraints[j][1][l,1][1])
                    Q = zeros(T,delta,delta)
                    for k=1:blockinfo.n_samples[j]
                        cur_a = k+(s-1+div(r*(r-1),2))*blockinfo.n_samples[j]
                        for rnk=1:blockinfo.ranks[j][l]
                            v = constraints[j][1][l,k][rnk] #v_{j,l,k,i}
                            sgn = constraints[j][4][l,k][rnk]
                            #the Q which should be added to the j,l,(r,s) and (s,r) block of P
                            #a[i]*v*v' is not always symmetric, a[i]*(v*v') is.
                            Q += Symmetric(sgn/T(2)*a[cur_a+j_idx]*(v*v'))
                        end
                        cur_a+=1 # tuple (r,s,k)
                    end
                    if r != s
                        # initial_matrix.blocks[j].blocks[l][(r-1)*delta+1:r*delta, (s-1)*delta+1:s*delta] += Q
                        initial_matrix.blocks[j].blocks[l][(s-1)*delta+1:s*delta, (r-1)*delta+1:r*delta] += Q #Q = Q'
                    else
                        initial_matrix.blocks[j].blocks[l][(r-1)*delta+1:r*delta, (r-1)*delta+1:r*delta] += 2*Q #Q = Q'
                    end
                end
            end
            initial_matrix.blocks[j].blocks[l] .= Symmetric(initial_matrix.blocks[j].blocks[l])
        end
    end
    return nothing #initial matrix modified, nothing returned
end

"""Compute the search directions, using a precomputed decomposition"""
function compute_search_direction(constraints,P,p,d,R,X_inv,Y,blockinfo,(C,CinvB,Q))
    # using the decomposition, compute the search directions
    # 5) solve system with rhs dx,dy <- (-d - Tr(A_* Z) ;  p) with Z = X^{-1}(PY - R)
    # 6) compute dX = P + sum_i A_i dx_i
    # 7) compute dY = X^{-1}(R-dX Y) (XdY = R-dXY)
    # 8) symmetrize dY = 1/2 (dY +dY')
    time_Z = @elapsed begin
        Z = similar(Y)
        Threads.@threads for j=1:blockinfo.J
            Z.blocks[j].blocks .= X_inv.blocks[j].blocks .* (P.blocks[j].blocks .* Y.blocks[j].blocks .- R.blocks[j].blocks)
            Z.blocks[j].blocks .= 1 ./T(2) .*(Z.blocks[j].blocks .+ transpose.(Z.blocks[j].blocks)) #the fix. Only necessary when m[j]>1
        end
    end

    rhs_y = p
    time_rhs_x = @elapsed begin
        rhs_x = -d - trace_A(constraints,Z,blockinfo)
    end

    # solve the system (C 0; CinvB^T I)(I 0; 0 LL^T)(C^T -CinvB; 0 I)(dx; dy) = (rhs_x; rhs_y)
    indices = blockinfo.x_indices #0, dim_S[1], dim_S[1]+dim_S[2],... ,sum(dim_S)
    time_sys = @elapsed begin
        #first lower triangular system:
        temp_x = similar(rhs_x)
        temp_y = [similar(rhs_y) for j=1:blockinfo.J]
        Threads.@threads for j=1:blockinfo.J
            temp_x[indices[j]+1:indices[j+1]] = C[j].L \ rhs_x[indices[j]+1:indices[j+1]]
            temp_y[j] = CinvB[j]' * temp_x[indices[j]+1:indices[j+1]]
        end
        temp_y = rhs_y - sum(temp_y)

        #second system: temp_x stays the same
        dy  = Q \ temp_y

        #third system:
        dx = similar(rhs_x)
        Threads.@threads for j=1:blockinfo.J
            dx[indices[j]+1:indices[j+1]] = C[j].U \ (temp_x[indices[j]+1:indices[j+1]] + CinvB[j] * dy)
        end
    end #of timing system

    #step 6:
    time_dX = @elapsed begin
        dX = I*P # making sure dX does not reference to P anymore
        add_weighted_A!(dX,constraints,dx,blockinfo)
    end

    #step 7 & 8
    time_dY = @elapsed begin
        dY = similar(Y)
        Threads.@threads for j=1:blockinfo.J
            dY.blocks[j].blocks .= X_inv.blocks[j].blocks .* (R.blocks[j].blocks .- dX.blocks[j].blocks .* Y.blocks[j].blocks)
            dY.blocks[j].blocks .= 1 ./T(2) .*(dY.blocks[j].blocks .+ transpose.(dY.blocks[j].blocks)) #symmetrize
        end
    end

    return dx,dX,dy,dY, [time_Z,time_rhs_x,time_sys,time_dX,time_dY]
end


"""Compute the step length min(γ α(M,dM), 1), where α is the maximum number step
to which keeps M+α(M,dM) dM positive semidefinite"""
function compute_step_length(M::BlockDiagonal,dM::BlockDiagonal,gamma)
    # per block:
    chol = [cholesky.(m.blocks) for m in blocks(M)]
    min_eig = [T(Inf) for j=1:length(chol)]
    Threads.@threads for j in 1:length(chol)
        for l in 1:length(chol[j])
            #LML should be symmetric. Not always in last few digits, so symmetrizing it
            #eigmin(Symmetric(LML)) throws an error
            LML = chol[j][l].L \ dM.blocks[j].blocks[l] / chol[j][l].U
            min_eig[j] = min(eigmin(1/T(2)*(LML+LML')), min_eig[j])
        end
    end
    min_eig = min(min_eig...)

    if min_eig > -gamma
        return T(1)
    else
        return -gamma/T(min_eig)
    end
end


end #of module MPMP
