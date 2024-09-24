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

#pragma once

#include <cstdio>
#include <stdexcept>
#include <vector>
#include <functional>

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include<iostream>
//using namespace std

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

template <typename InType, typename OutType = InType, typename ComputeType = OutType>
struct TestBench {
    using SampleRunner = std::function<void()>;

    TestBench(size_t m, size_t n, size_t k, bool use_bias, cublasOperation_t transa = CUBLAS_OP_T, cublasOperation_t transb = CUBLAS_OP_N,
            ComputeType alpha = ComputeType{0}, ComputeType beta = ComputeType{0},
            size_t workspaceSize = 0/*1024 * 1024 * 32 or 4*/, size_t N = 1,
            ComputeType Ascale = ComputeType{1}, ComputeType Bscale = ComputeType{1/*0.5*/},
            ComputeType Cscale = ComputeType{1}, ComputeType Dscale = ComputeType{1}) :
        m(m), n(n), k(k), N(N), use_bias(use_bias), alpha(alpha), beta(beta), workspaceSize(workspaceSize), Ahost(m * k * N), Bhost(n * k * N),
        Chost(m * n * N), biasHost(m * N), AscaleHost(Ascale), BscaleHost(Bscale), CscaleHost(Cscale), DscaleHost(Dscale),
		transA(transa), transB(transb) {
        checkCublasStatus(cublasLtCreate(&ltHandle));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Adev), m * k * N * sizeof(InType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Bdev), n * k * N  * sizeof(InType)));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Cdev), m * n * N  * sizeof(OutType)));
        //std::cout << "use_bias: " << use_bias << std::endl;
        if(use_bias)
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&biasDev), m * N * sizeof(OutType)));
        checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
        checkCudaStatus(cudaStreamCreate(&stream));

        // Currently only fp8 supports per-tensor scaling
        perTensorScalingEnabled = std::is_same<InType, __nv_fp8_e4m3>::value || std::is_same<InType, __nv_fp8_e5m2>::value;

        if (perTensorScalingEnabled) {
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&AscaleDev), sizeof(*AscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&BscaleDev), sizeof(*BscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&CscaleDev), sizeof(*CscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&DscaleDev), sizeof(*DscaleDev)));
            checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&DamaxDev), sizeof(*DamaxDev)));
        }

		lda = (transa == CUBLAS_OP_N) ? m : k;
		ldb = (transb == CUBLAS_OP_N) ? k : n;
		ldc = m;
		ldd = m;
		lde = m;
        //std::cout << "input  size: "<< sizeof(InType) << std::endl;
        //std::cout << "output size: "<< sizeof(OutType) << std::endl;
        //std::cout << "comput size: "<< sizeof(ComputeType) << std::endl;
        fillData(use_bias);
    }

    ~TestBench() {
        checkCublasStatus(cublasLtDestroy(ltHandle));
        checkCudaStatus(cudaFree(Adev));
        checkCudaStatus(cudaFree(Bdev));
        checkCudaStatus(cudaFree(Cdev));
        if(use_bias)
            checkCudaStatus(cudaFree(biasDev));
        checkCudaStatus(cudaFree(workspace));
        if (perTensorScalingEnabled) {
            checkCudaStatus(cudaFree(AscaleDev));
            checkCudaStatus(cudaFree(BscaleDev));
            checkCudaStatus(cudaFree(CscaleDev));
            checkCudaStatus(cudaFree(DscaleDev));
            checkCudaStatus(cudaFree(DamaxDev));
        }
        checkCudaStatus(cudaStreamDestroy(stream));
    }

	template <typename T>
	void hipblaslt_init_sin(
	    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
	{
	    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
	#pragma omp parallel for
	        for(size_t j = 0; j < N; ++j)
	        {
	            size_t offset = j * lda + i_batch * stride;
	            for(size_t i = 0; i < M; ++i)
	                A[i + offset] = static_cast<T>(sin(double(i + offset))); //force cast to double
	        }
	}

	template <typename T>
	void hipblaslt_init_cos(
	    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
	{
	    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
	#pragma omp parallel for
	        for(size_t j = 0; j < N; ++j)
	        {
	            size_t offset = j * lda + i_batch * stride;
	            for(size_t i = 0; i < M; ++i)
	                A[i + offset] = T(cos(double(i + offset))); //force cast to double
	        }
	}

    void fillData(bool use_bias) {
		size_t A_row = (transA == CUBLAS_OP_N) ? m : k;
		size_t A_col = (transA == CUBLAS_OP_N) ? k : m;
		size_t B_row = (transB == CUBLAS_OP_N) ? k : n;
        size_t B_col = (transB == CUBLAS_OP_N) ? n : k;
		size_t stride_a = lda * A_col;
		size_t stride_b = ldb * B_col;
		size_t stride_c = ldc * n;
		size_t stride_d = ldd * n;
		size_t stride_e = lde * n;
		size_t sizeA = stride_a;
		size_t sizeB = stride_b;
		size_t sizeC = stride_c;
		size_t sizeD = stride_d;
		size_t sizeE = stride_e;

		hipblaslt_init_sin<InType>(Ahost.data(), A_row, A_col, lda, stride_a);
		hipblaslt_init_cos<InType>(Bhost.data(), B_row, B_col, ldb, stride_b);

        // for (int i = 0; i < m * k * N; i++) Ahost[i] = InType(i);
        // for (int i = 0; i < n * k * N; i++) Bhost[i] = InType(i);
        
        if(use_bias)
            for (int i = 0; i < m * N; i++) biasHost[i] = InType(i + 1.0);
    }

    void copyDataToDevice() {
        checkCudaStatus(cudaMemcpyAsync(Adev, Ahost.data(), Ahost.size() * sizeof(Ahost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(cudaMemcpyAsync(Bdev, Bhost.data(), Bhost.size() * sizeof(Bhost[0]), cudaMemcpyHostToDevice, stream));
        if(use_bias)
            checkCudaStatus(cudaMemcpyAsync(biasDev, biasHost.data(), biasHost.size() * sizeof(biasHost[0]), cudaMemcpyHostToDevice));
        if (perTensorScalingEnabled) {
            checkCudaStatus(cudaMemcpyAsync(AscaleDev, &AscaleHost, sizeof(AscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(BscaleDev, &BscaleHost, sizeof(BscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(CscaleDev, &CscaleHost, sizeof(CscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(DscaleDev, &DscaleHost, sizeof(DscaleHost), cudaMemcpyHostToDevice));
            checkCudaStatus(cudaMemcpyAsync(DamaxDev, &DamaxHost, sizeof(DamaxHost), cudaMemcpyHostToDevice));
        }
    }

    void copyDataFromDevice() {
        checkCudaStatus(cudaMemcpyAsync(Chost.data(), Cdev, Chost.size() * sizeof(Chost[0]), cudaMemcpyDeviceToHost, stream));
    }

    void streamSynchronize() {
        checkCudaStatus(cudaStreamSynchronize(stream));
    }

    void run(const SampleRunner& runSample) {
        copyDataToDevice();

        runSample();

        copyDataFromDevice();
        streamSynchronize();
    }

    bool perTensorScalingEnabled;
    size_t m, n, k, N;
    bool use_bias;
    ComputeType alpha, beta;
    size_t workspaceSize;
    std::vector<InType> Ahost, Bhost;
    std::vector<OutType> Chost, biasHost;
    void *workspace;
    InType *Adev, *Bdev;
    OutType *Cdev, *biasDev;
    cudaStream_t stream;
    cublasLtHandle_t ltHandle;
    ComputeType AscaleHost, BscaleHost, CscaleHost, DscaleHost, DamaxHost;
    ComputeType *AscaleDev, *BscaleDev, *CscaleDev, *DscaleDev, *DamaxDev;
	cublasOperation_t transA, transB;
	size_t lda, ldb, ldc, ldd, lde;
};
//__nv_half
//__half
template <>
inline void TestBench<__nv_half, __nv_half, float>::fillData(bool use_bias) {
    for (int i = 0; i < m * k * N; i++) Ahost[i] = __float2half_rn(i);
    for (int i = 0; i < n * k * N; i++) Bhost[i] = __float2half_rn(i);
    if(use_bias)
        for (int i = 0; i < m * N; i++) biasHost[i] = __float2half_rn(i + 1);
}

template <>
inline void TestBench<__nv_half, float, float>::fillData(bool use_bias) {
    for (int i = 0; i < m * k * N; i++) Ahost[i] = __float2half_rn(i);
    for (int i = 0; i < n * k * N; i++) Bhost[i] = __float2half_rn(i);
    if(use_bias)
        for (int i = 0; i < m * N; i++) biasHost[i] = __float2half_rn(i + 1);
}

template <>
inline void TestBench<__half, __half, cuComplex>::fillData(bool use_bias) {
    for (int i = 0; i < m * k * N; i++) Ahost[i] = __float2half_rn(i/100.);
    for (int i = 0; i < n * k * N; i++) Bhost[i] = __float2half_rn(i/100.);
    if(use_bias)
        for (int i = 0; i < m * N; i++) biasHost[i] = __float2half_rn(i + 1);
}
