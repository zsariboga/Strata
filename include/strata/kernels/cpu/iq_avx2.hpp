// include/strata/kernels/cpu/iq_avx2.hpp - AVX-2 multi-token dot products for the i-quant expert
// formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS) against Q8_K activations (ggml's block_q8_K).
//
// The AVX-512 kernel's scheme on the CPUs without it (AMD Zen 2/3, Intel Core 12th-14th gen): the weights of a
// 32-value chunk are decoded once per verify window and every token applies them with five instructions.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels::cpu {

bool iq256_supported(int ggml_type) noexcept;
/// Bit-identical to ggml's singleton gate/up for IQ2_XXS, IQ2_XS, IQ3_XXS and IQ3_S.
/// The caller checks AVX2 support and passes one Q8_K activation.
void iq256_gu_rows_exact_one(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                             const void* act, float* ff, int r0, int r1);

/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]), rows [r0, r1); gate rows at blob, up rows at blob + up_off.
void iq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
/// out[t][r] = w_r . a[t], rows [r0, r1).
void iq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                float* const* out, int r0, int r1);

/// The kernels' variants, a mask.  Every variant computes the same bits; they differ in speed from core to core.
/// kIq256Gather: IQ3_XXS, IQ3_S and IQ2_S read their grid entries with one gather (STRATA_IQ256_GATHER).
/// kIq256Vnni: AVX-VNNI's vpdpwssd / vpdpbusd, every format and iq4nl256_down_rows (cpu_avxvnni_ok).
inline constexpr int kIq256Gather = 1;
inline constexpr int kIq256Vnni = 2;
/// The variant iq256_gu_rows / iq256_rows take on the calling thread.
int iq256_variant() noexcept;
/// Every variant bit this build and this CPU can run (tests and benches).
int iq256_variants() noexcept;
/// iq256_gu_rows / iq256_rows in a given variant (tests and benches; only bits of iq256_variants()).
void iq256_gu_rows_v(int variant, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                     const void* const* act, int nt, float* const* ff, int r0, int r1);
void iq256_rows_v(int variant, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act,
                  int nt, float* const* out, int r0, int r1);

/// ggml's quantize_row_q8_K (x86 runs the scalar reference), byte-identical, in AVX-2: n values -> n/256 block_q8_K.
void q8k_quant_avx2(const float* x, void* y, int64_t n);

/// out[t][r] = w_r . h[t] for IQ4_NL (type 20) rows against Q8_0 activations (ggml's block_q8_0).
/// IQ4_NL is a 32-value-block format, so this does not go through iq256_rows (QK_K blocks, Q8_K acts).
void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                        float* const* out, int r0, int r1);
/// The same in a given variant (only the kIq256Vnni bit matters here).
void iq4nl256_down_rows_v(int variant, const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt,
                          float* const* out, int r0, int r1);

// ---- local STRATA_CPU_REPACK_MIB (opt-in): a CPU copy of an expert's gate/up rows in a 4-bit code layout.
// Every 32-value half of an IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S block keeps the decode's own magnitude bytes,
// signs and scales (one signed 4-bit code per weight: a magnitude index | sign << 3, and two int8 scales per half), so the
// kernels below sum the same integers in the same order and the rows come out bit for bit as the original kernels'.
// Layout per 256 values: the block's fp16 d (2 B), 8 x 2 scale bytes, 8 x 16 code bytes (low nibbles values 0-15 of a
// half, high nibbles 16-31) = 146 B.  The GPU keeps reading the original blob: this is an extra copy in RAM.
constexpr int kRepackBlockBytes = 146;
/// Whether `ggml_type`'s gate/up rows can be repacked (IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ3_S 21, IQ2_S 22).
bool iq256_repack_supported(int ggml_type) noexcept;
/// Repacks `rows` gate/up rows of `n` values (gate rows at `blob`, up rows at `blob + up_off`, `gu_row` bytes apart)
/// into `out` (rows * (n / 256) * kRepackBlockBytes bytes: the gate rows, then the up rows).  False (with `err`)
/// when a half does not fit the layout - then nothing may be registered for this blob.
bool iq256_repack_gu(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int rows, int n, uint8_t* out,
                     std::string& err);
/// Registers a repacked copy for the original blob pointer (before the pool runs; read-only afterwards).
void cpu_repack_register(const uint8_t* blob, const uint8_t* repacked);
/// The repacked copy registered for `blob`, or nullptr.
const uint8_t* cpu_repack_find(const uint8_t* blob) noexcept;
/// The repacked rows in the order of iq256_gu_rows (`rows_half`: gate rows; up rows follow), any token count.
void iq256_gu_rows_rp(int ggml_type, const uint8_t* rp, int rows_half, int n, const void* const* act, int nt,
                      float* const* ff, int r0, int r1);
/// The same as iq256_gu_rows_exact_one (one token, ggml's float order), from the repacked rows.
void iq256_gu_rows_rp_exact_one(int ggml_type, const uint8_t* rp, int rows_half, int n, const void* act, float* ff,
                                int r0, int r1);

}  // namespace strata::kernels::cpu
