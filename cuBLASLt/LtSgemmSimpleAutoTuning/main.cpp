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

#include <cstdio>
#include <vector>
#include <string>
#include<iostream>

#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "sample_cublasLt_LtSgemmSimpleAutoTuning.h"
#include "helpers.h"

using namespace std;


void printAlgo(const cublasLtMatmulAlgo_t& algo) {
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme;

    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK, sizeof(numSplitsK), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme, sizeof(reductionScheme), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption), NULL));

    printf("algo={ Id=%d, tileIdx=%d splitK=%d reduc=%d swizzle=%d custom=%d }\n",
        algoId, tile, numSplitsK, reductionScheme, swizzle, customOption);
}

// m, n, k, transA, transB, requested_solution, num_cold_iters, num_hot_iters
int main(int argc, char* argv[]) {
	if (argc != 11) {
		printf("[ERROR] argument number must be 10.");
		return 1;
	}
	size_t m = static_cast<size_t>(stoi(argv[1]));
	size_t n = static_cast<size_t>(stoi(argv[2]));
	size_t k = static_cast<size_t>(stoi(argv[3]));
	string transA_str = argv[4];
	string transB_str = argv[5];
	cublasOperation_t transA = (transA_str == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t transB = (transB_str == "N") ? CUBLAS_OP_N : CUBLAS_OP_T;
    int requested_num = stoi(argv[6]);
    int num_cold_iters = stoi(argv[7]);
    int num_hot_iters = stoi(argv[8]);
    string type = argv[9];
    bool use_bias = (string(argv[10]) == "use_bias");
    //std::cout << "use_bias: " << use_bias << endl;

    //TestBench<__nv_bfloat16, __nv_bfloat16, float> props(m, n, k, transA, transB, 1.0f, 0.0f, 1024 * 1024 * 4);
    if(type == "hss"){
        TestBench<__nv_half, float, float> props(m, n, k, use_bias, transA, transB, 1.0f, 0.0f, 8320 * 2048 * 4);
        cout << transA_str << "," << transB_str << "," << props.m << "," << props.n << "," << props.k << ",";

        cublasLtMatmulAlgo_t algo;
        props.run([&props, &algo, &requested_num, &num_cold_iters, &num_hot_iters, &type, &use_bias] {
            LtSgemmSimpleAutoTuning(props.ltHandle,
                                    props.transA,
                                    props.transB,
                                    props.m,
                                    props.n,
                                    props.k,
                                    &props.alpha,
                                    props.Adev,
                                    props.lda,
                                    props.Bdev,
                                    props.ldb,
                                    &props.beta,
                                    props.Cdev,
                                    props.ldc,
                                    props.biasDev, /*add device bias vector*/
                                    props.workspace,
                                    0/*props.workspaceSize*/,
                                    algo,
                                    requested_num,
                                    num_cold_iters,
                                    num_hot_iters,
                                    type,
                                    use_bias);
        });
        // printAlgo(algo);
    }
    else if(type == "hhs"){
        TestBench<__nv_half, __nv_half, float> props(m, n, k, use_bias, transA, transB, 1.0f, 1.5f, 8320 * 2048 * 4);

        cout << transA_str << "," << transB_str << "," << props.m << "," << props.n << "," << props.k << ",";

        cublasLtMatmulAlgo_t algo;
        props.run([&props, &algo, &requested_num, &num_cold_iters, &num_hot_iters, &type, &use_bias] {
            LtSgemmSimpleAutoTuning(props.ltHandle,
                                    props.transA,
                                    props.transB,
                                    props.m,
                                    props.n,
                                    props.k,
                                    &props.alpha,
                                    props.Adev,
                                    props.lda,
                                    props.Bdev,
                                    props.ldb,
                                    &props.beta,
                                    props.Cdev,
                                    props.ldc,
                                    props.biasDev, /*add device bias vector*/
                                    props.workspace,
                                    0/*props.workspaceSize*/,
                                    algo,
                                    requested_num,
                                    num_cold_iters,
                                    num_hot_iters,
                                    type,
                                    use_bias);
        });
        // printAlgo(algo);
    }
    else{
		std::cout << "type error: " << type << std::endl;
		return 1;
    }

    return 0;
}
