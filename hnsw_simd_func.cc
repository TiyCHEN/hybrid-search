#pragma once

#include <immintrin.h>
#include <iostream>
#include <stddef.h>
#include "hnsw_simd_dist_func.h"
#include "hnswlib/hnswlib.h"
SIMDFuncType SIMDFunc = nullptr;
// using SIMDFuncType = float (*)(const float *, const float *, int);
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

#if defined(__AVX512F__)
export float F32L2AVX512(const void *pv1_, const void *pv2_,const void *dim_) {
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t dim = *((float*)(dim_));
    size_t dim16 = dim >> 4;
    const float* pv1 = (float*)(pv1_);
    const float* pv2 = (float*)(pv2_);
    const float *pEnd1 = pv1 + (dim16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pv1 < pEnd1) {
        v1 = _mm512_loadu_ps(pv1);
        pv1 += 16;
        v2 = _mm512_loadu_ps(pv2);
        pv2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] +
                TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}

void SetSIMDFunc(){
    // std::cout<<"Set F32L2AVX512"<<std::endl;
    // SIMDFunc = F32L2AVX512;
    SIMDFunc = base_hnsw::HybridSimd;
}


#elif defined(__AVX__)
export float F32L2AVX(const void *pv1_, const void *pv2_, const void* dim_) {
    // std::cout<<dim<<std::endl;
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t dim = *((float*)(dim_));
    size_t dim16 = dim >> 4;
    const float* pv1 = (float*)(pv1_);
    const float* pv2 = (float*)(pv2_);
    const float *pEnd1 = pv1 + (dim16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pv1 < pEnd1) {
        v1 = _mm256_loadu_ps(pv1);
        pv1 += 8;
        v2 = _mm256_loadu_ps(pv2);
        pv2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pv1);
        pv1 += 8;
        v2 = _mm256_loadu_ps(pv2);
        pv2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

void SetSIMDFunc(){
    // std::cout<<"Set F32L2AVX"<<std::endl;
    // SIMDFunc = F32L2AVX;
    SIMDFunc = base_hnsw::HybridSimd;
}

#else
    float F32L2BF(const void *pv1_, const void *pv2_, const void* dim_) {
        float res = 0;
        const float* pv1 = (float*)(pv1_);
        const float* pv2 = (float*)(pv2_);
        size_t dim = *((float*)(dim_));
        for (size_t i = 0; i < dim; i++) {
            float t = pv1[i] - pv2[i];
            res += t * t;
        }
        return res;
    }

    void SetSIMDFunc(){
        // std::cout<<"Set F32L2BF"<<std::endl;
        SIMDFunc = F32L2BF;
    }
#endif