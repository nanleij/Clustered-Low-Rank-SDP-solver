module SpherePacking

using AbstractAlgebra
using SpecialFunctions
using MPMP
using WriteFilesSDPB

export testbound_sphere_packing, Nsphere_packing_2point

function spherevolume(n, r)
    sqrt(BigFloat(pi))^n / gamma(BigFloat(n)/2 + 1) * r^n
end


laguerre(k, alpha, x) = MPMP.laguerrebasis(k, alpha, x)[end]

function standard_basis(N, i, j, element = 1, symmetric = true)
    # symmetric: place element on both i,j and j,i (i = j => element on i,i)
    # non-symmetric: place element on i,j
    E = zeros(parent(element), N, N)
    E[i,j] = element
    if symmetric
        E[j,i] = element
    end
    return E
end

function Nsphere_packing_2point(n, d, r, N=2, file_path="", write_only=false; omega = BigFloat(10)^2, normalize_prep = false,kwargs...)
    if precision(BigFloat) == 256 #default, usually need at least 512
        setprecision(512)
    end
    #NOTE: we use 2d as max degree, and sort of 4d because we use u = ||x||^2 as variable
    #variables M, a_{ij,k} : for k = 0:d for i = 1:N for j=1:i
    # max -M
    # M - f_ii(0)>=0 for i=1:N
    # F(f)(0) - (vol B(r_i)^1/2 vol B(r_j)^1/2 )_{ij=1}^N >=0
    # F(f)(t) >=0 for t>=0
    # -f(w)_ij >= 0 for w>=r_i+r_j for i=1:N j=1:i (symmetric)
    # With f(x) = sum_k a_k k!/pi^k L_k^{n/2-1}(pi ||x||^2)
    # and thus F(f)(x) = sum_k a_k x^{k}
    # max -M
    # s.t.  - (vol B(r_i)^1/2 vol B(r_j)^1/2 )_{ij=1}^N + \sum_i \sum_j a_{ij,k} E_ij >= 0 (G={1}) (NxN)
    #       0 + sum_k sum_i sum_j a_{ij,k} E_{ij} x^k >= 0        (G = {1,x}) (NxN)
    #       0 - sum_k a_ijk k! pi^-k L_k^{n/2-1}(pi x) >=0          (G = {1,x - (r_i + r_j)^2}) (1x1)
    #       M - sum_k a_iik k!/pi^k L_k^{n/2-1}(0) >= 0 for i=1:N   G = {1}, 1x1

    #Initialize polynomial rings for the polynomial constraints
    F = RealField
    R, (x,) = PolynomialRing(F,["x"])
    SN = MatrixSpace(R,N,N)
    S1 = MatrixSpace(R,1,1) # for the constraints on individual elements

    # The constraint matrices
    #order: constant matrix, matrix for M, matrices for a_ijk: for k=0:2d for i=1:N for j=1:i
    # - (vol B(r_i)^1/2 vol B(r_j)^1/2 )_{ij=1}^N + \sum_i \sum_j a_{ij,k} E_ij >= 0 (G={1}) (NxN)
    M0 = [vcat(SN([R(-sqrt(spherevolume(n,r[i])*spherevolume(n,r[j]))) for i=1:N, j=1:N]), SN(R(0)),
            [ k == 0 ? SN(standard_basis(N,i,j,R(1),true)) : SN(R(0)) for k=0:2d for i=1:N for j=1:i])]
    #    0 + sum_k sum_i sum_j a_{ij,k} E_{ij} x^k >= 0        (G = {1,x}) (NxN)
    M1 = [vcat(SN(R(0)), SN(R(0)),[SN(standard_basis(N,i,j,x^k,true)) for k=0:2d for i=1:N for j=1:i])]
    #    0 - sum_k a_ijk k! pi^-k L_k^{n/2-1}(pi x) >=0       i=1:N, j=1:i   (G = {1,x - (r_i + r_j)^2}) (1x1)
    M2 = [[vcat(S1(R(0)), S1(R(0)),
        [ (r == i && s == j) ? S1(-factorial(big(k)) / BigFloat(pi)^k * laguerre(k, BigFloat(n)/2-1, BigFloat(pi)*x)) : S1(R(0)) for k=0:2d for r=1:N for s=1:r])] for i=1:N for j=1:i]
    #    0 + M - sum_k a_iik k!/pi^k L_k^{n/2-1}(0) >= 0 for i=1:N   G = {1}, 1x1
    M3 = [[vcat(S1(R(0)), S1(R(1)),
        [ r == s == i ? S1(-factorial(big(k)) / BigFloat(pi)^k * laguerre(k, BigFloat(n)/2-1, BigFloat(0))) : S1(R(0)) for k=0:2d for r=1:N for s=1:r])] for i=1:N]
    M = vcat(M0,M1,M2...,M3...) #M2 and M3 contain multiple constraints

    #sample points
    sample_points = vcat([BigFloat(0)],
      [BigFloat.(MPMP.create_sample_points_1d(2d))], #laguerre sample points
      [BigFloat.(MPMP.create_sample_points_1d(2d)).+ (r[i]+r[j])^2 for i=1:N for j=1:i],
      [[BigFloat(0)] for i=1:N])

    #polynomial weights
    G = vcat([[R(1)]],
        [[R(1),x]],
        [[R(1),x-(r[i]+r[j])^2] for i=1:N for j=1:i],
        [[R(1)] for i=1:N])

    # bases
    q = MPMP.laguerrebasis(d, BigFloat(n)/2-1, BigFloat(2*pi)*x)
    max_coef = [max(coeffs(q[i])...) for i=1:length(q)]
    q = [max_coef[i]^(-1)*q[i] for i=1:length(q)]

    #maximum degrees
    delta = vcat(0,2d,[2d for i=1:N for j=1:i],[0 for i=1:N])

    #objective = -M
    b = vcat(BigFloat(-1),[BigFloat(0) for k=0:2d for i=1:N for j=1:i])

    #get the constraints and blockinfo
    constraints = [prepareabc(M[j],G[j],q,sample_points[j],delta[j];normalize=normalize_prep) for j=1:length(G)]
    blockinfo = MPMP.get_block_info(constraints)

    if length(file_path)>0
        #if needed, write the constraints to files in SDPB format
        write_files(file_path,constraints,blockinfo,b)
    end
    if length(M) == 7#only change ordering if N = 2, and hence J = 7.
        #ordering of sdpb: pairs (3,6),(5,7), (4,1) and (2) on cores (if 4 cores).
        # In any case, not the polynomial constraints 2,3,4,5 on the same core if multiple cores
        ordering = [3,6,5,7,4,1,2]
        constraints = constraints[ordering]
        blockinfo = MPMP.get_block_info(constraints)
    end

    if !write_only
        #solve the MPMP and return the values.
        # default start value is 100. For large n,d this may need to be increased.
        x,X,y,Y,P,p,d,dual_gap,primal_obj,dual_obj = solverank1sdp(constraints, b, blockinfo;omega_p = omega,omega_d = omega, kwargs...);
    else
        return true
    end
end

function test_bound_sphere_packing(n=3,d=8)
    if precision(BigFloat) == 256
        setprecision(512)
    end

    #best lower bound known with these ratio is obtained from the crystal structure of NaCl
    cur_bound = Nsphere_packing_2point(n,d,[BigFloat(1), sqrt(BigFloat(2))-1],2)
    println(-cur_bound[end])

    NaCl_density = 0.793
    NaCl_bound = 0.813 #this is the bound reported by de Laat et al. in Upper bounds for packings of spheres of several radii.
    #The same bound is programmed by Nsphere_packing_2point, but then with sampling instead of coefficient matching.
    println("Compare to the density of NaCL: $NaCl_density (Current bound: $NaCl_bound)")
end

end # end module
