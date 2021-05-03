# Clustered-Low-Rank-SDP-solver
An SDP solver exploiting structures of clusters of constraints and low-rank of the constraint matrices. This is a more general version of SDPB (TODO: add link), and part of the MSc thesis (TODO: add link).

(TODO: add program it solves)

# Functionality
Functions include 
- `solvempmp`, which converts multivariate polynomial programs to a clustered low-rank SDP and calls the solver. 
- `prepareabc`, which does the conversion for a single constraint.
- functions to create sample points and bases

Using multiple threads is supported. For small problems, this might not give a speed up due to the overhead. To use multiple cores, start Julia with  `julia -t [number of threads]`.

# Dependencies
- `Arblib`, for arbitrary precision arithmetic
- `BlockDiagonals`
- `GenericSVD`
- `GenericLinearAlgebra`

# References
TODO: add main references
