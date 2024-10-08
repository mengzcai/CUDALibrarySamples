/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <string>
/// Sample wrapper executing single precision gemm algorithm auto tuning by querying cublasLt heuristics for best algorithms,
/// iterate over the results and pick the algorithm that have the best performance for the given problem
///
/// pointer mode is always host, to change it configure the appropriate matmul descriptor attribute
/// matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to change
/// this configure appropriate attribute in the preference handle
//template <typename __nv_bfloat16, typename __nv_bfloat16/* = __nv_bfloat16*/, typename float/*= OutType*/>
//void LtSgemmSimpleAutoTuning(cublasLtHandle_t ltHandle,
//                             cublasOperation_t transa,
//                             cublasOperation_t transb,
//                             int m,
//                             int n,
//                             int k,
//                             const __nv_bfloat16 *alpha, /* host pointer */
//                             const __nv_bfloat16 *A,
//                             int lda,
//                             const __nv_bfloat16 *B,
//                             int ldb,
//                             const __nv_bfloat16 *beta, /* host pointer */
//                             __nv_bfloat16 *C,
//                             int ldc,
//                             void *workspace,
//                             size_t workspaceSize,
//                             cublasLtMatmulAlgo_t& algo,
//							 const int requested_num,
//							 const int num_cold_iters,
//							 const int num_hot_iters);
//

//#define gen(Tin, Tout, Tc) 
//template<typename Tin, typename Tout, typename Tc>

/*template<typename Tin, typename Tout, typename Tc>
    void LtSgemmSimpleAutoTuning(cublasLtHandle_t ltHandle,\
                             cublasOperation_t transa,\
                             cublasOperation_t transb,\
                             size_t m,\
                             size_t n,\
                             size_t k,\
                             const Tc *alpha, \
                             const Tin *A,\
                             size_t lda,\
                             const Tin *B,\
                             size_t ldb,\
                             const Tc *beta, \
                             Tout *C,\
                             size_t ldc,\
                             void *workspace,\
                             size_t workspaceSize,\
                             cublasLtMatmulAlgo_t& algo,\
                             const int requested_num,\
                             const int num_cold_iters,\
                             const int num_hot_iters,\
                             std::string type);*/

void LtSgemmSimpleAutoTuning(cublasLtHandle_t ltHandle,\
                            cublasOperation_t transa,\
                            cublasOperation_t transb,\
                            size_t m,\
                            size_t n,\
                            size_t k,\
                            const float *alpha, \
                            const __nv_half *A,\
                            size_t lda,\
                            const __nv_half *B,\
                            size_t ldb,\
                            const float *beta, \
                            void *C,\
                            size_t ldc,\
                            void *biasDev, /*add device bias vector*/
                            void *workspace,\
                            size_t workspaceSize,\
                            cublasLtMatmulAlgo_t& algo,\
                            const int requested_num,\
                            const int num_cold_iters,\
                            const int num_hot_iters,\
                            std::string type,
                            bool use_bias);


