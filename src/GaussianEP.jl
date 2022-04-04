module GaussianEP

export block_expectation_propagation, expectation_propagation, Term, TermRBM, EPState, EPOut
export Prior, IntervalPrior, SpikeSlabPrior, BinaryPrior, GaussianPrior, PosteriorPrior, QuadraturePrior, AutoPrior, ThetaPrior, UniformPrior, ReLUPrior

using MKL, ExtractMacro, SpecialFunctions, LinearAlgebra, DelimitedFiles, Random
using FastGaussQuadrature, ForwardDiff

include("Term.jl")
include("priors.jl")
include("expectation_propagation.jl")
include("ProgressReporter.jl")
include("Analytics.jl")

end # end module
