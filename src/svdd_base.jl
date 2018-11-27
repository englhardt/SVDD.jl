using MLKernels
using JuMP
using MLLabelUtils
using StatsBase
using NearestNeighbors
import MLBase

const learning_pool_enc = LabelEnc.NativeLabels([:U, :Lin, :Lout])
const class_label_enc = LabelEnc.NativeLabels([:inlier, :outlier])
const OPT_PRECISION = 1e-7

@enum ModelState model_created=1 model_initialized=2 model_fitted=3

const Scope = Union{Val{:Subspace}, Val{:Global}}

abstract type ModelException <: Exception end

struct ModelStateException <: ModelException
    actual::ModelState
    expected::ModelState
end

Base.showerror(io::IO, e::ModelStateException) = print(io, "Invalid model state: $(e.actual). Model must at least be in state $(e.expected).")

struct ModelInvariantException <: ModelException
    msg
end

Base.showerror(io::IO, e::ModelInvariantException) = print(io, "Model invariant error. $(e.msg)")

abstract type OCClassifier end

abstract type SVDDClassifier <: OCClassifier end

abstract type SubOCClassifier <: OCClassifier end

abstract type InitializationStrategy end
