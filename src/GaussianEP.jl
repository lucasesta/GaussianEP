module GaussianEP

export expectation_propagation, Term, TermRBM, EPState, EPOut, BlockEPState
export Prior, IntervalPrior, SpikeSlabPrior, BinaryPrior, GaussianPrior, PosteriorPrior, QuadraturePrior, AutoPrior, ThetaPrior, UniformPrior, ReLUPrior

using ExtractMacro, SpecialFunctions, LinearAlgebra, DelimitedFiles, Random
using FastGaussQuadrature, ForwardDiff

include("Term.jl")
include("priors.jl")
include("expectation_propagation.jl")
include("ProgressReporter.jl")
include("Analytics.jl")

end # end module
