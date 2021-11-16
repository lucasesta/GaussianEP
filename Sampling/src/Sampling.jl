# In this package we implement several MC strategies for estimating
# the moments of a RBM visibile and hidden variables, depending
# on the specific choice of the prior.

"""

    E(v,h) = -∑_iμ v_i * w_iμ *h_μ + ∑_μ U_μ(h_μ) + ∑_i V_i(v_i)
    P(v,h) ∝ exp[-E(v,h)]

"""


module Sampling
using Statistics
using BenchmarkTools
using LinearAlgebra
using FastGaussQuadrature, ForwardDiff

include("priors.jl")
include("mc_sampling.jl")
include("pre_process.jl")
include("utils.jl")
end
