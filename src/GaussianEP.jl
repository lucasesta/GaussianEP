module GaussianEP

export expectation_propagation, Term, TermRBM, EPState, EPOut
export Prior, IntervalPrior, SpikeSlabPrior, BinaryPrior, GaussianPrior, PosteriorPrior, QuadraturePrior, AutoPrior, ThetaPrior, UniformPrior

using ExtractMacro, SpecialFunctions, LinearAlgebra, DelimitedFiles

include("Term.jl")
include("priors.jl")
include("expectation_propagation.jl")
include("ProgressReporter.jl")
include("../MC/MC_rbm.jl")
include("Analytics.jl")

end # end module
