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
#include <algorithm>
#include <string>

#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "sample_cublasLt_LtSgemmSimpleAutoTuning.h"
#include "helpers.h"

float median(std::vector<float>& times) {
    const size_t size = times.size();
    if (size == 0) {
        return 0;
    }
	printf("median: ");
	for (int i = 0; i < times.size(); i++) {
		printf("%f, ", times[i]);
	}
	printf("\n");
    std::sort(times.begin(), times.end());

    const size_t mid = size / 2;
    if (size % 2 == 0) {
        return (times[mid] + times[mid - 1]) / 2;
    }
    else {
        return times[mid];
    }
}

/// Sample wrapper executing single precision gemm algorithm auto tuning by querying cublasLt heuristics for best algorithms,
/// iterate over the results and pick the algorithm that have the best performance for the given problem
///
/// pointer mode is always host, to change it configure the appropriate matmul descriptor attribute
/// matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to change
/// this configure appropriate attribute in the preference handle
//template <typename InType, typename OutType/* = InType*/, typename ComputeType/* = OutType*/>
void LtSgemmSimpleAutoTuning(cublasLtHandle_t ltHandle,
                             cublasOperation_t transa,
                             cublasOperation_t transb,
                             size_t m,
                             size_t n,
                             size_t k,
                             const float *alpha, /* host pointer */
                             const __nv_half *A,
                             size_t lda,
                             const __nv_half *B,
                             size_t ldb,
                             const float *beta, /* host pointer */
                             void *C,
                             size_t ldc,
                             void *biasDev, /*add device bias vector*/
                             void *workspace,
                             size_t workspaceSize,
                             cublasLtMatmulAlgo_t& algo,
                             const int requested_num,
                             const int num_cold_iters,
                             const int num_hot_iters,
                             std::string type,
                             bool use_bias) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    const int requestedAlgoCount = requested_num;
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult[requestedAlgoCount] = { 0 };
    int bestAlgoIdx = 0;
    float time = 0;
    float bestAlgoTime = 0;
    cudaStream_t stream;
    cudaEvent_t startEvent, stopEvent;

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    cudaDataType output_type = CUDA_R_32F;
    if(type == "hss"){
        // create matrix descriptors, we are good with the details here so no need to set any extra attributes
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));
        output_type=CUDA_R_32F;
    }
    else if(type == "hhs"){
        // create matrix descriptors, we are good with the details here so no need to set any extra attributes
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
        checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, ldc));
        output_type=CUDA_R_16F;
    }
    else{
        std::cout << "error type : " << type << std::endl;
        return;
    }

    // Condition for device bias vector
    if(use_bias){
        // create epilogue, add Bias vector for comparison
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        checkCublasStatus(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

        cudaDataType bias_data_type = output_type;
        checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(cudaDataType)));
        // create bias pointer and setting it
        checkCublasStatus(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasDev, sizeof(void*)));
    }

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize, sizeof(workspaceSize)));

//    printf("workspaceSize = %d\n", workspaceSize);
    //printf("CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = %d\n", CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES);

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference,
        requestedAlgoCount, heuristicResult, &returnedResults));
    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    workspaceSize = 0;
    for(int i = 0; i < returnedResults; i++)
        workspaceSize = std::max(workspaceSize, heuristicResult[i].workspaceSize);

//    printf("after: workspaceSize = %d\n", workspaceSize);
    checkCudaStatus(cudaStreamCreate(&stream));
    checkCudaStatus(cudaEventCreate(&startEvent));
    checkCudaStatus(cudaEventCreate(&stopEvent));

    // constexpr int repeatAlgoCheck = 5;  // nuhot_call
    // std::vector<float> algoTimes(repeatAlgoCheck);

    for (int algoIdx = 0; algoIdx < returnedResults; algoIdx++) {
        printf("[MENGZCAI] algoIdx = %d\n", algoIdx);
		for (int cold_num = 0; cold_num < num_cold_iters; cold_num++) {
            if(type == "hss"){
                checkCublasStatus(cublasLtMatmul(ltHandle,
                                                operationDesc,
                                                alpha,
                                                A,
                                                Adesc,
                                                B,
                                                Bdesc,
                                                beta,
                                                static_cast<float*>(C),
                                                Cdesc,
                                                static_cast<float*>(C),
                                                Cdesc,
                                                &heuristicResult[algoIdx].algo,
                                                workspace,
                                                workspaceSize,
                                                stream));
		        //std::cout << "type : " << type << std::endl;
            }
            else if(type == "hhs"){
                checkCublasStatus(cublasLtMatmul(ltHandle,
                                                operationDesc,
                                                alpha,
                                                A,
                                                Adesc,
                                                B,
                                                Bdesc,
                                                beta,
                                                static_cast<__nv_half*>(C),
                                                Cdesc,
                                                static_cast<__nv_half*>(C),
                                                Cdesc,
                                                &heuristicResult[algoIdx].algo,
                                                workspace,
                                                workspaceSize,
                                                stream));
		        //std::cout << "type : " << type << std::endl;
            }
		}
		checkCudaStatus(cudaEventRecord(startEvent, stream));
        for (int checkIdx = 0; checkIdx < num_hot_iters; checkIdx++) {
            // checkCudaStatus(cudaEventRecord(startEvent, stream));

            if(type == "hss"){
                checkCublasStatus(cublasLtMatmul(ltHandle,
                                                operationDesc,
                                                alpha,
                                                A,
                                                Adesc,
                                                B,
                                                Bdesc,
                                                beta,
                                                static_cast<float*>(C),
                                                Cdesc,
                                                static_cast<float*>(C),
                                                Cdesc,
                                                &heuristicResult[algoIdx].algo,
                                                workspace,
                                                workspaceSize,
                                                stream));
		        //std::cout << "type : " << type << std::endl;
            }
            else if(type == "hhs"){
                checkCublasStatus(cublasLtMatmul(ltHandle,
                                                operationDesc,
                                                alpha,
                                                A,
                                                Adesc,
                                                B,
                                                Bdesc,
                                                beta,
                                                static_cast<__nv_half*>(C),
                                                Cdesc,
                                                static_cast<__nv_half*>(C),
                                                Cdesc,
                                                &heuristicResult[algoIdx].algo,
                                                workspace,
                                                workspaceSize,
                                                stream));
		        //std::cout << "type : " << type << std::endl;
            }

            // checkCudaStatus(cudaEventRecord(stopEvent, stream));
            // checkCudaStatus(cudaEventSynchronize(stopEvent));
            // checkCudaStatus(cudaEventElapsedTime(&time, startEvent, stopEvent));
            // algoTimes[checkIdx] = time;
        }

	    checkCudaStatus(cudaEventRecord(stopEvent, stream));
        checkCudaStatus(cudaEventSynchronize(stopEvent));
        checkCudaStatus(cudaEventElapsedTime(&time, startEvent, stopEvent));

        // time = median(algoTimes);
		time /= static_cast<float>(num_hot_iters);
        // printf("algo_idx: %d, %f\n", algoIdx, time * 1000);
        if (algoIdx == 0 || time < bestAlgoTime) {
            bestAlgoTime = time;
            bestAlgoIdx = algoIdx;
        }
        printf("[MENGZCAI] algoIdx %d end------------\n", algoIdx);
    }

    memcpy(&algo, &heuristicResult[bestAlgoIdx].algo, sizeof(algo));
	// printf("%d solutions are supported.\n", returnedResults);
	// printf("bestAlgoTime: %f\n", bestAlgoTime * 1000);
    printf("%f,%d\n", bestAlgoTime * 1000, bestAlgoIdx);
    // printf("%f,%d\n", bestAlgoTime * 1000,returnedResults);

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
    if (stream) checkCudaStatus(cudaStreamDestroy(stream));
    if (startEvent) checkCudaStatus(cudaEventDestroy(startEvent));
    if (stopEvent) checkCudaStatus(cudaEventDestroy(stopEvent));
}
