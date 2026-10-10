// include/strata/kernels/route_prior.hpp - STRATA_ROUTE_PRIOR=lambda (opt-in, changes the output): cache-conditioned
// routing in verify windows.  See src/kernels/cuda/route_prior.cu.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Top-10 per token over `logits` (n_tok x 512) where a resident expert (`res_layer[e] >= 0`, the layer's residency
/// table) gets `lambda` added to its logit for the SELECTION; the weights are the router's own probabilities
/// (softmax over the 512 unbiased logits) renormalised over the 10 picked, as the native router does.
void route_prior_top10(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights,
                       int n_tok, void* stream);

/// local fixes (both opt-in; set once at start-up, before any window graph is captured):
///  - `gate` (STRATA_ROUTE_PRIOR_SCOPE): a mapped float the serve loop writes per window; the kernel reads lambda from
///    it, and at <= 0 writes nothing, so the native router's top-10 already in ids/weights stands (bitwise PRIOR off).
///  - `protect` (STRATA_ROUTE_PRIOR_PROTECT=N): the router's own first N picks are never displaced; the bonus only
///    reorders ranks N..9.
///  - `keep_p` (STRATA_ROUTE_PRIOR_KEEP_P=x): the router's picks whose probability is at least x of its top pick's are
///    never displaced either (an adaptive rank: more protected when the router is sure).
void route_prior_configure(const float* gate, int protect, float keep_p);

}  // namespace strata::kernels
