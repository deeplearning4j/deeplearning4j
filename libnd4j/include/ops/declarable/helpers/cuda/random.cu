/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/random.h>
//#include <NativeOps.h>
#include <vector>
#include <memory>
#include <graph/Context.h>
#include <helpers/RandomLauncher.h>
#include <helpers/ShapeUtils.h>
#include <array/NDArrayFactory.h>
#include <exceptions/cuda_exception.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>

namespace sd {
namespace ops {
namespace helpers {
    /**
     * gammaLess - compute gamma distributed value for shapes (alpha) from 0 to 1
     * @tparam T - any float types are acceptable
     * @param U - uniform random generated vals
     * @param alpha - shape of distribution
     * @param beta - scale of distributed values
     * @return gamma distributed value
     */
    template <typename T>
    T __device__ gammaLess(T const* U, Nd4jLong index, Nd4jLong maxLength, T const alpha, T const beta) {
        auto d = T(1.0334f) - T(0.0766f) * math::p_exp(T(2.2942f) * alpha);
        auto a = math::p_pow(T(2.f), alpha) * math::p_pow(T(1.f) - math::p_exp(-d * T(0.5f)), alpha);
        auto b = alpha * math::p_pow(d, alpha - T(1.f)) * exp(-d);
        auto c = a + b;
        T rawX;
        auto indexV = index;
        auto underAlpha = T(1.f) / alpha;
        auto powerAlpha = math::p_pow(T(2.f), alpha - T(1.f));

        for (;;) {
            auto u = (indexV < maxLength)?U[indexV++]:U[0];
              if (indexV >= maxLength) indexV = 0LL;
//            math::atomics::nd4j_atomicAdd(index, 1LL);
            if (u <= a / c) rawX = -T(2.f) * math::p_log(T(1.f) - T(0.5f) * math::p_pow(c * u, underAlpha));
            else            rawX = - math::p_log(c * (T(1.f) - u)/(alpha * math::p_pow(d, alpha - T(1.f))));

            T v = indexV < maxLength?U[indexV++]:U[0];
            if (indexV >= maxLength) indexV = 0LL;
//            math::atomics::nd4j_atomicAdd(index, 1LL);

            if (rawX <= d) {
                auto testVal = (math::p_pow(rawX, alpha - 1.f) * math::p_exp(-T(0.5f) * rawX)) / (powerAlpha * math::p_pow(T(1.f) - math::p_exp(-T(0.5f) * rawX), alpha - T(1.f)));
                if (testVal < v) continue;
                break;
            }
            else {
                if (v <= math::p_pow(d / rawX, T(1.f) - alpha)) break;
                continue;
            }
        }
        return rawX / beta;
    }

    /**
     * gammaGreat - generate gamma distributed value for shape (alpha) greater then 1
     * @tparam T - given type (any float type is accepted.)
     * @param rng  - random generator
     * @param alpha - shape of the gamma distribution (alpha)
     * @param beta  - scale of the gamma distribution (beta)
     * @return - gamma distributed value with given params
     */
    template <typename T>
    T __device__ gammaGreat(T const* U, Nd4jLong index, Nd4jLong maxLength, T const alpha, T const beta) {
        auto decreasedAlpha = alpha - T(1.f/3.f);
        auto c = T(1.)/ math::p_sqrt(T(9.f) * decreasedAlpha);
//        static auto index = 0LL;
        auto indexV = index;
        T x;
        auto normalDistributed = [U, maxLength](Nd4jLong& index) {
            auto v1 = index < maxLength?U[index++]:U[0];
            if (index >= maxLength) index = 0LL;
//            math::atomics::nd4j_atomicAdd(index, 1LL);
            auto v2 = index < maxLength?U[index++]:U[0];
            if (index >= maxLength) index = 0LL;
//            math::atomics::nd4j_atomicAdd(index, 1LL);

            return math::p_cos(T(2.f * 3.141592f) * v2) * math::p_sqrt(T(-2.f) * math::p_log(v1));
        };

        float normalizedVar;
        for(;;) {
            do {
                x = normalDistributed(indexV); //printf("X = %f\n", x);
                normalizedVar = T(1.f) + c * x;
            } while(normalizedVar < T(0.f));
            normalizedVar = normalizedVar * normalizedVar * normalizedVar; //v * v * v;

            auto u = U[indexV++];
            if (indexV >= maxLength) indexV = 0LL;
//            math::atomics::nd4j_atomicAdd(index, 1LL);

            if( u < T(1.f) - T(.0331f) * (x * x) * (x * x) )
                break; //return (d * v / b);
            if( log(u) < 0.5f * x * x + decreasedAlpha * (1. - normalizedVar + math::p_log(normalizedVar)) )
                break;
        }
        return (decreasedAlpha * normalizedVar / beta);
    }

    /*
     * fillGammaKernel - fill up output with gamma distributed values
     *
     *  uList - uniformly distributed values set
     *  uLength - length of uList
     *  alpha - alpha param
     *  beta - beta param
     *  output - distributed output.
     * */
    template <typename T>
    static __global__ void fillGammaKernel(T const* uList, Nd4jLong uLength, T const* alpha, const Nd4jLong* alphaShape,
            T const* beta, const Nd4jLong* betaShape, T* output, const Nd4jLong* outputShape) {
        // fill up
        __shared__ Nd4jLong aLength;
        __shared__ Nd4jLong outLength;
        if (threadIdx.x == 0) {
            aLength = shape::length(alphaShape);
            outLength = shape::length(outputShape) / aLength;
        }
        __syncthreads();

        for (auto k = blockIdx.x; k < (int)outLength; k += gridDim.x) {
            auto pos = k * aLength;
//            auto u = uList[k]; // this is a vector
            //Nd4jLong index = k;
            for (auto e = threadIdx.x; e < (int)aLength; e += blockDim.x) {
                auto aIndex = shape::getIndexOffset(e, alphaShape);
                auto bIndex = betaShape?shape::getIndexOffset(e, betaShape):-1LL;
                auto betaV = T(beta != nullptr ? beta[bIndex] : T(1.f));
                auto zIndex = shape::getIndexOffset(e + pos, outputShape);

                output[zIndex] = alpha[aIndex] > T(1.f)?gammaGreat(uList, pos, uLength, alpha[aIndex], betaV):gammaLess(uList, pos, uLength, alpha[aIndex], betaV);
            }
        }
    }

    template <typename T>
    static void fillRandomGamma_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output) {
        // To fill up output need to broadcast alpha and beta to the same shape and in
        const Nd4jLong* broadcasted = nullptr;
        if (beta != nullptr)
            ShapeUtils::evalBroadcastShapeInfo(*alpha, *beta, true, broadcasted, context->getWorkspace());
        else
            broadcasted = alpha->shapeInfo();
        auto step = shape::length(broadcasted);
        auto shift = output->lengthOf() * 4LL; // 2-wise greater case for uniform vals

        auto copyAlpha = alpha;
        auto copyBeta = beta;
        if (beta != nullptr) {
            NDArray alphaBroadcasted(broadcasted, alpha->dataType(), true, context);
            NDArray betaBroadcasted(broadcasted, beta->dataType(), true, context);

            copyAlpha = new NDArray(alphaBroadcasted.applyTrueBroadcast(BroadcastOpsTuple::Assign(), *alpha));
            copyBeta = new NDArray(betaBroadcasted.applyTrueBroadcast(BroadcastOpsTuple::Assign(), *beta));
//            if (!copyAlpha->isActualOnDevice()) copyAlpha->syncToDevice();
//            if (!copyBeta->isActualOnDevice()) copyBeta->syncToDevice();
        }

        auto stream = context->getCudaStream();
        NDArray uniform = NDArrayFactory::create<T>('c', {shift}, context);
        uniform.syncToDevice();
        // fill up uniform with given length
        RandomLauncher::fillUniform(context, rng, &uniform, 0.0000000001, 0.9999999999);
        uniform.syncToDevice();
//        uniform.printIndexedBuffer("Uniform");
        fillGammaKernel<T><<<128, 128, 256, *stream>>>(uniform.dataBuffer()->specialAsT<T>(), shift,
                copyAlpha->dataBuffer()->specialAsT<T>(), copyAlpha->specialShapeInfo(),
                beta?copyBeta->dataBuffer()->specialAsT<T>():(T const*)nullptr,
                beta?copyBeta->specialShapeInfo():(Nd4jLong const*)nullptr,
                output->dataBuffer()->specialAsT<T>(), output->specialShapeInfo());

        if (beta != nullptr) {
            delete copyAlpha;
            delete copyBeta;
            //delete broadcasted;
        }

    }

    void fillRandomGamma(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output) {
        if (beta)
            NDArray::prepareSpecialUse({output}, {alpha, beta});
        else
            NDArray::prepareSpecialUse({output}, {alpha});
        BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomGamma_, (context, rng, alpha, beta, output), FLOAT_NATIVE);
        if (beta)
            NDArray::registerSpecialUse({output}, {alpha, beta});
        else
            NDArray::prepareSpecialUse({output}, {alpha});
    }
    BUILD_SINGLE_TEMPLATE(template void fillRandomGamma_, (LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta, NDArray* output), FLOAT_NATIVE);


    /*
     * algorithm Poisson generator based upon the inversion by sequential search
     *
    init:
         Let x ← 0, p ← e−λ, s ← p.
         using uniformly random sequence U (u in U) distributed at [0, 1].
    while u > s do:
         x ← x + 1.
         p ← p * λ / x.
         s ← s + p.
    return x.
     * */
    template <typename T>
    static __global__ void fillPoissonKernel(T* uList, Nd4jLong uLength, T* lambda, const Nd4jLong* lambdaShape,
                                             T* output, const Nd4jLong* outputShape) {

        __shared__ Nd4jLong step;

        if (threadIdx.x == 0) {
            step = shape::length(lambdaShape);
        }
        __syncthreads();

        for (auto k = blockIdx.x; k < (int)uLength; k += gridDim.x) {
            auto pos = k * step;
            auto u = uList[k];
            for (auto e = threadIdx.x; e < step; e += blockDim.x) {
                auto p = math::nd4j_exp<T,T>(-lambda[e]);
                auto s = p;
                auto x = T(0.f);
                auto lIndex = shape::getIndexOffset(e, lambdaShape);
                auto zIndex = shape::getIndexOffset(e + pos, outputShape);
                while (u > s) {
                    x += T(1.);
                    p *= lambda[lIndex] / x;
                    s += p;
                }
                output[zIndex] = x;
            }
        }
    }

    template <typename T>
    static void fillRandomPoisson_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
        auto shift = output->lengthOf() / lambda->lengthOf();
        NDArray uniform('c', {shift}, output->dataType());
        auto stream = context->getCudaStream();
        // fill up uniform with given length
        RandomLauncher::fillUniform(context, rng, &uniform, 0., 1.);
        fillPoissonKernel<T><<<128, 256, 128, *stream>>>(uniform.dataBuffer()->specialAsT<T>(), uniform.lengthOf(),
                lambda->dataBuffer()->specialAsT<T>(), lambda->specialShapeInfo(),
                output->dataBuffer()->specialAsT<T>(), output->specialShapeInfo());
    }

    void fillRandomPoisson(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {lambda});
        BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomPoisson_, (context, rng, lambda, output), FLOAT_NATIVE);
        NDArray::registerSpecialUse({output}, {lambda});
    }

    BUILD_SINGLE_TEMPLATE(template void fillRandomPoisson_, (LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda, NDArray* output), FLOAT_NATIVE);

    template <typename T>
    static __global__ void fillUniformKernel(graph::RandomGenerator* devRng, T from, T to, T* output, const Nd4jLong* outputShape) {
        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;

        __shared__ Nd4jLong outputLen;

        if (0 == threadIdx.x) {
            outputLen = shape::length(outputShape);
        }
        __syncthreads();

        for (auto i = start; i < outputLen; i += step) {
            auto zIndex = shape::getIndexOffset(i, outputShape);
            output[zIndex] = devRng->relativeT<T>(i, from, to);
        }

    }

    template <typename T>
    static void fillRandomUniform_(LaunchContext* context, graph::RandomGenerator& rng, NDArray* min, NDArray* max, NDArray* output) {
        T minVal = T(0);
        T maxVal = DataTypeUtils::infOrMax<T>();
        if (min)
            minVal = min->t<T>(0);
        if (max)
            maxVal = max->t<T>(0);

        if (output->isR())
            RandomLauncher::fillUniform(context, rng, output, minVal, maxVal);
        else {
            auto stream = context->getCudaStream();
            graph::RandomGenerator *devRng;
            auto err = cudaMalloc(&devRng, sizeof(graph::RandomGenerator));
            if (err != 0) {
                cuda_exception::build("fillRandomUniform_: Cannot allocate device memory for random generator due error", err);
            }

            err = cudaMemcpy(devRng, &rng, sizeof(graph::RandomGenerator), cudaMemcpyHostToDevice);
            if (err != 0) {
                cuda_exception::build("fillRandomUniform_: Cannot copy random generator to device", err);
            }
            auto outputBuf = output->dataBuffer()->specialAsT<T>();
            auto outputShape = output->specialShapeInfo();
            fillUniformKernel<T><<<128, 128, 128, *stream>>>(devRng, minVal, maxVal, outputBuf, outputShape);

            err = cudaStreamSynchronize(*stream);
            if (err != 0) {
                cuda_exception::build("fillRandomUniform_: Cannot successfully finish kernel call", err);
            }

            err = cudaFree(devRng);
            if (err != 0) {
                cuda_exception::build("fillRandomUniform_: Cannot deallocate device memory for random generator", err);
            }
        }
    }

    void fillRandomUniform(LaunchContext* context, graph::RandomGenerator& rng, NDArray* min, NDArray* max, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), fillRandomUniform_, (context, rng, min, max, output), NUMERIC_TYPES);
    }

///////////////////////////////////////////////////////////////////
// used https://en.wikipedia.org/wiki/Categorical_distribution
// methods: gumbel trick + softmax + argmax
template<typename X, typename Z>
__global__ static void fillMultiNomialCuda_(graph::RandomGenerator* devRng, const void* vx, const Nd4jLong* xShapeInfo, 
                                     void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong batchValue, 
                                     const Nd4jLong numOfSamples, const Nd4jLong numOfClassX, 
                                     const Nd4jLong dimA, const X minVal, const X maxVal) {
                                  
    
    const X* x = reinterpret_cast<const X*>(vx);
    Z* z = reinterpret_cast<Z*>(vz);
   
    __shared__ Nd4jLong xDimAstride, zDimAstride, xDimCstride, zDimCstride, dimC;

    if (0 == threadIdx.x) {
        dimC = (0 == dimA) ? 1 : 0;
        zDimAstride = shape::stride(zShapeInfo)[dimA];
        xDimAstride = shape::stride(xShapeInfo)[dimA];
        zDimCstride = shape::stride(zShapeInfo)[dimC];
        xDimCstride = shape::stride(xShapeInfo)[dimC];
    }
    __syncthreads();

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (Nd4jLong index = tid; index < batchValue*numOfSamples; index += gridDim.x * blockDim.x) {
        
        Nd4jLong nBatchIndex = index / numOfSamples;
        Nd4jLong nSampleIndexInBatch = index - (nBatchIndex * numOfSamples);
        
        const X* xTad = x + (nBatchIndex * xDimCstride);
        Z* zTad = z + (nBatchIndex * zDimCstride);
        Z& arg = zTad[nSampleIndexInBatch * zDimAstride];
        
        X Max = -minVal;
        Nd4jLong nSamplesPerBatch = nBatchIndex * numOfClassX * numOfSamples;
        Nd4jLong nClassPerSamples = nSampleIndexInBatch * numOfClassX;
        
        for (Nd4jLong nClass = 0; nClass < numOfClassX; nClass++) {
            Nd4jLong nIndex = nSamplesPerBatch + nClassPerSamples + nClass;
            X tValue = (xTad[nClass * xDimAstride] - sd::math::nd4j_log<X, X>(-sd::math::nd4j_log<X, X>(devRng->relativeT<X>(nIndex, minVal, maxVal))));
            if (tValue > Max) {
                Max = tValue; 
                arg = nClass;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename X, typename Z>
__host__ static void fillMultiNomialCudaLauncher(
    const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream,
    graph::RandomGenerator* devRng, const void* vx, const Nd4jLong* xShapeInfo, 
    void* vz, const Nd4jLong* zShapeInfo, 
    const Nd4jLong batchValue, const Nd4jLong numOfSamples, 
    const Nd4jLong numOfClassX, const Nd4jLong dimA){

    const X minVal = DataTypeUtils::min<X>();
    const X maxVal = 1.0;
    
    fillMultiNomialCuda_<X, Z> <<< blocksPerGrid, threadsPerBlock, 256, * stream >>> (
        devRng, vx, xShapeInfo, vz, zShapeInfo, batchValue,
        numOfSamples, numOfClassX, dimA, minVal, maxVal);
}
 
///////////////////////////////////////////////////////////////////
void fillRandomMultiNomial(LaunchContext* context, graph::RandomGenerator& rng, NDArray& input, NDArray& output, const Nd4jLong numOfSamples, const int dimC) {

     Nd4jLong dimA = (0 == dimC) ? 1 : 0;

     const Nd4jLong batchValue = output.sizeAt(dimC);
     const Nd4jLong numOfClassX = input.sizeAt(dimA);
     
     const int threadsPerBlock = MAX_NUM_THREADS / 2;
     const int blocksPerGrid = (batchValue * numOfSamples + threadsPerBlock - 1) / threadsPerBlock;
    
     PointersManager manager(context, "fillMultinomial");
     graph::RandomGenerator *devRng;

     auto err = cudaMalloc(&devRng, sizeof(graph::RandomGenerator));
     if (err != 0) {
         cuda_exception::build("fillRandomMultiNomial: Cannot allocate device memory for random generator due error", err);
     }
     err = cudaStreamSynchronize(*context->getCudaStream());
     if (err != 0) {
         cuda_exception::build("fillRandomMultiNomial: Cannot synchronize stream for random generator due error", err);
     }
     err = cudaMemcpyAsync(devRng, &rng, sizeof(graph::RandomGenerator), cudaMemcpyHostToDevice, *context->getCudaStream());
     if (err != 0) {
         cuda_exception::build("fillRandomMultiNomial: Cannot copy random generator to device", err);
     }

     NDArray::prepareSpecialUse({ &output }, { &input });
     BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), fillMultiNomialCudaLauncher, 
      (blocksPerGrid, threadsPerBlock, context->getCudaStream(), devRng, input.specialBuffer(),
       input.specialShapeInfo(), output.specialBuffer(),
       output.specialShapeInfo(), batchValue, numOfSamples, 
       numOfClassX, dimA), FLOAT_TYPES, INDEXING_TYPES);
     NDArray::registerSpecialUse({ &output }, { &input });
     manager.synchronize();

     err = cudaFree(devRng);
     if (err != 0) {
         cuda_exception::build("fillRandomMultiNomial: Cannot deallocate device memory for random generator", err);
     }
     rng.rewindH(output.lengthOf() * numOfClassX);
 }

}
}
}