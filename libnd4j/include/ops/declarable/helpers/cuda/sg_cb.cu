/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
//

#include <ops/declarable/helpers/sg_cb.h>
#include <exceptions/cuda_exception.h>
#include <array/NDArrayFactory.h>

#define HS_MAX_EXP 6.0f

namespace sd {
    namespace ops {
        namespace helpers {
            template <typename T>
            __global__ void hSoftmaxKernel(void *vsyn0, void *vsyn1, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code, int expLength, bool isInference) {

                auto syn0 = reinterpret_cast<T*>(vsyn0);
                auto syn1 = reinterpret_cast<T*>(vsyn1);
                auto expTable = reinterpret_cast<T*>(vexpTable);
                auto neu1e = reinterpret_cast<T*>(vneu1e);

                T dot(0.0f);
                T g(0.0f);
                T f(0.0f);

                // dot
                for (int e = 0; e < vectorLength; e++) {
                    dot += syn0[e] * syn1[e];
                }

                // gradient
                if (dot < (T) - HS_MAX_EXP || dot >= (T) HS_MAX_EXP)
                    return;


                int idx = static_cast<int>((dot + HS_MAX_EXP) * ((float) expLength / HS_MAX_EXP / 2.0f));

                if (idx >= expLength || idx < 0)
                    return;

                f = expTable[idx];
                g = (static_cast<T>(1.0f) - static_cast<T>(code) - f) * (T) alpha;

                // axpy1

                for (int e = 0; e < vectorLength; e++) {
                    neu1e[e] = g * syn1[e] + neu1e[e];
                }

                // axpy2
                if (!isInference) {
                    for (int e = 0; e < vectorLength; e++) {
                        syn1[e] = g * syn0[e] + syn1[e];
                    }
                }
            }

            template <typename T>
            void hSoftmax_(void *vsyn0, void *vsyn1, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code, int expLength, bool isInference, cudaStream_t* stream) {
                hSoftmaxKernel<T><<<1,1,128, *stream>>>(vsyn0, vsyn1, vexpTable, vneu1e, alpha, vectorLength, code, expLength, isInference);
            }

            template <typename T>
            __global__ void nSamplingKernel(void *vsyn0, void *vsyn1Neg, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code, int expLength, bool isInference) {
                auto syn0 = reinterpret_cast<T*>(vsyn0);
                auto syn1Neg = reinterpret_cast<T*>(vsyn1Neg);
                auto expTable = reinterpret_cast<T*>(vexpTable);
                auto neu1e = reinterpret_cast<T*>(vneu1e);

                T dot = (T) 0.0f;
                T g = (T) 0.0f;

                for (int e = 0; e < vectorLength; e++) {
                    dot += syn0[e] * syn1Neg[e];
                }

                if (dot > HS_MAX_EXP)
                    g = (code - 1) * alpha;
                else if (dot < (T) - HS_MAX_EXP)
                    g = (code - 0) * alpha;
                else {
                    int idx = (int) ((dot + (T) HS_MAX_EXP) * ((T) expLength / HS_MAX_EXP / 2.0));
                    if (idx >= expLength)
                        return;

                    if (idx < 0)
                        return;

                    g = ((T) code - expTable[idx]) * alpha;
                }

                // axpy1
                for (int e = 0; e < vectorLength; e++) {
                    neu1e[e] = g * syn1Neg[e] + neu1e[e];
                }

                // axpy2
                if (!isInference) {
                    for (int e = 0; e < vectorLength; e++) {
                        syn1Neg[e] = g * syn0[e] + syn1Neg[e];
                    }
                }
            }

            template <typename T>
            void nSampling_(void *vsyn0, void *vsyn1Neg, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code, int expLength, bool isInference, cudaStream_t* stream) {
                nSamplingKernel<T><<<1,1,128, *stream>>>(vsyn0, vsyn1Neg, vexpTable, vneu1e, alpha, vectorLength, code, expLength, isInference);
            }

            /*
             * binarySearch - find element in haystack buffer (haystack - sorted device memory)
             * */
            int binarySearch(const int *haystack, const int needle, const int totalElements) {
                int firstIndex = 0;
                int lastIndex = totalElements - 1;
                int halfIndex = sd::math::nd4j_floor<float, int>((lastIndex + firstIndex) / (float) 2);

                while(haystack[halfIndex] != needle && firstIndex < lastIndex) {
                    if (needle < haystack[halfIndex]) {
                        lastIndex = halfIndex - 1;
                    } else if (needle > haystack[halfIndex]) {
                        firstIndex = halfIndex + 1;
                    }
                    halfIndex = sd::math::nd4j_floor<float, int>((lastIndex + firstIndex) / (float) 2);
                }

                return (haystack[halfIndex] == needle) ? halfIndex : -1;
            }
            template <typename T>
            __global__ void addInfVectorKernel(T* neu1, T* infVector, int vectorLength) {
                auto start = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = blockDim.x * gridDim.x;

                for (auto i = start; i < vectorLength; i += step) {
                    neu1[i] += infVector[i];
                }
            }

            template <typename T>
            void skipgram_(NDArray& s0, NDArray& s1, NDArray& s1n, NDArray& expTableV, NDArray& negTableV, NDArray& infV, int target, int ngStarter, NDArray& indices, NDArray& codes, double alpha, Nd4jLong randomValue, const int hsRounds, const int nsRounds) {
//                    void *vsyn0, void *vsyn1, void *vsyn1Neg, void *vexpTable, void *vnegTable, void *vinfVector, int target, int ngStarter, int *indices, int8_t *codes, double alpha, Nd4jLong randomValue, const int hsRounds, const int nsRounds, const int vocabSize, const int vectorLength, const int expLength, const int negLength) {
                auto syn0 = reinterpret_cast<T*>(s0.specialBuffer());
                auto syn1 = reinterpret_cast<T*>(s1.specialBuffer());
                auto syn1Neg = reinterpret_cast<T*>(s1n.specialBuffer());
                auto expTable = reinterpret_cast<T*>(expTableV.specialBuffer());
                auto negTable = reinterpret_cast<T*>(negTableV.specialBuffer());
                auto infVector = reinterpret_cast<T*>(infV.specialBuffer());
                const int vocabSize = s0.sizeAt(0);
                const int vectorLength = s0.sizeAt(1);
                const int expLength = expTableV.lengthOf();
                const int negLength = negTableV.lengthOf();
                indices.tickReadDevice();
                indices.syncToHost();
                codes.tickReadDevice();
                codes.syncToHost();
                auto stream = s0.getContext()->getCudaStream();

                T* neu1e; // = new T[vectorLength];
                //memset(neu1e, 0, vectorLength * sizeof(T));
                auto err = cudaMalloc(&neu1e, sizeof(T) * vectorLength);
                err = cudaMemset(neu1e, 0, sizeof(T) * vectorLength);
                // hierarchic softmax goes first (if enabled)

                auto syn0row = infVector != nullptr ? infVector : syn0 + (target * vectorLength);
                auto irow = 0;
                if (hsRounds > 0) {
                    for (int r = 0; r < hsRounds; r++) {
                        irow = indices.t<int>(r);
                        if (irow < 0 || irow >= vocabSize)
                            break;

                        hSoftmax_<T>(syn0row, syn1 + (irow * vectorLength), expTable, neu1e, alpha, vectorLength, codes.t<int8_t>(r), expLength, infVector != nullptr, stream);
                    }
                }

                // negative sampling goes second (if enabled)
                auto nsStarter = ngStarter;
                irow = nsStarter;
                if (nsRounds > 0) {
                    for (int r = 0; r < nsRounds + 1; r++) {
                        if (r == 0) {
                            // target is known in advance
                        } else {
                            randomValue = randomValue * (unsigned long long) 25214903917 + 11;
                            auto idx = sd::math::nd4j_abs<Nd4jLong >((randomValue >> 16) % negLength);
                            irow = idx >= negLength ? -1 : negTableV.e<int>(idx);

                            if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;
                            if (irow == nsStarter)
                                continue;
                        }

                        nSampling_<T>(syn0row, syn1Neg + (irow * vectorLength), expTable, neu1e, alpha, vectorLength, r == 0 ? 1 : 0, expLength, infVector != nullptr, stream);
                    }
                }

                if (infVector == nullptr) {
                    addInfVectorKernel<T><<<128, 256, 256, *stream>>>(syn0row, neu1e, vectorLength);
                } else {
                    addInfVectorKernel<T><<<128, 256, 256, *stream>>>(infVector, neu1e, vectorLength);
                }
                err = cudaStreamSynchronize(*stream);
                if (0 != err) {
                    throw cuda_exception::build("helpers::skipgram_: Cannot synchronize stream after addInfVectorKernel", err);
                }

                err = cudaFree(neu1e);
                if (0 != err) {
                    throw cuda_exception::build("helpers::skipgram_: Cannot deallocate temp memory for lingual net", err);
                }
            }
            BUILD_SINGLE_TEMPLATE(template void skipgram_, (NDArray& syn0, NDArray& syn1, NDArray& syn1Neg, NDArray& expTable, NDArray& negTable, NDArray& infVector, int target, int ngStarter, NDArray& indices, NDArray& codes, double alpha, Nd4jLong randomValue, const int hsRounds, const int nsRounds), FLOAT_TYPES);

            /*
             * batched version of skipgram routine
             * */
            template <typename T>
            void skipgramBatchExec_(NDArray &s0, NDArray &s1, NDArray &s1n, NDArray& expTableV, NDArray& negTableV, NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr, NDArray &nextRandom, const int nsRounds, const bool preciseMode, const int numThreads) {
//            (NDArray &s0, NDArray &s1, NDArray &s1n, NDArray& expTable, NDArray& negTable, NDArray& infVector, NDArray& targets, NDArray& negStarters, NDArray& indices, NDArray& codes, NDArray& lr, NDArray& nextRandom, const int nsRounds, const bool preciseMode, const int numThreads) {
                //auto syn0 = reinterpret_cast<T*>(vsyn0);
                //auto syn1 = reinterpret_cast<T*>(vsyn1);
                //auto syn1Neg = reinterpret_cast<T*>(vsyn1Neg);
                auto stream = s0.getContext()->getCudaStream();
                negTableV.tickReadDevice();
                negTableV.syncToHost();
                const auto expTable = reinterpret_cast<T*>(expTableV.specialBuffer());
                const auto negTable = reinterpret_cast<T*>(negTableV.buffer());
                const auto infVector = (T*)nullptr; //reinterpret_cast<T*>(infVector.specialBuffer());

                const int vocabSize = s0.sizeAt(0);
                const int vectorLength = s0.sizeAt(1);
                const int expLength = expTableV.lengthOf();
                const int negLength = negTableV.lengthOf();

                //T sneu1e[600];

                //const auto numThreads = omp_get_max_threads();
                const auto idxShift = indices.isEmpty() ? 0 : indices.sizeAt(1);
                const auto hsRounds = codes.isEmpty() ? 0 : codes.sizeAt(1);

                // regular mode provides 0 guarantees for reproducibility
                auto numTargets = targets.lengthOf();
                targets.syncToHost();
                indices.syncToHost();
                codes.syncToHost();
                lr.syncToHost();
                nextRandom.syncToHost();
                negStarters.tickReadDevice();
                negStarters.syncToHost();
                auto bTarget = reinterpret_cast<int*>(targets.buffer()); //targets.bufferAsT<int>();
                auto bIndices = reinterpret_cast<int*>(indices.buffer()); //indices.bufferAsT<int>();
                auto bCodes = reinterpret_cast<int8_t*>(codes.buffer()); //codes.bufferAsT<int8_t>();

//                PRAGMA_OMP_PARALLEL_FOR_ARGS(num_threads(numThreads))
                for (int t = 0; t < numTargets; t++) {
                    T* neu1e;//lvectorLength <= 600 ? sneu1e : new T[vectorLength];
                    auto err = cudaMalloc(&neu1e, vectorLength * sizeof(T));
                    err = cudaMemset(neu1e, 0, vectorLength * sizeof(T));
                    //memset(neu1e, 0, vectorLength * sizeof(T));

                    auto target = bTarget[t];
                    auto alpha = lr.e<double>(t);
                    unsigned long long randomValue = nextRandom.e<Nd4jLong>(t);

                    auto syn0row = reinterpret_cast<T*>(s0.specialBuffer()) + (target * vectorLength);

                    if (hsRounds > 0) {
                        int irow = 0;
                        auto cShift = t * idxShift;

                        for (int e = 0; e < hsRounds; e++) {
                            irow = bIndices[e + cShift];
                            if (irow < 0 || irow >= vocabSize)
                                continue;

                            auto syn1row = reinterpret_cast<T*>(s1.getSpecialBuffer()) + (irow * vectorLength);
                            auto code = bCodes[e + cShift];

                            //nd4j_printf("syn0: [%i]; syn1: [%i]; code: [%i]\n", target, irow, code);
                            hSoftmax_<T>(syn0row, syn1row, expTable, neu1e, alpha, vectorLength, code, expLength, false, stream);
                        }
                    }


                    if (nsRounds > 0) {
                        int irow = negStarters.e<int>(t);
                        int nsStarter = irow;
                        for (int r = 0; r < nsRounds + 1; r++) {
                            if (r == 0) {
                                // target is known in advance
                            } else {
                                randomValue = randomValue * (unsigned long long) 25214903917 + 11;
                                auto idx = sd::math::nd4j_abs<Nd4jLong >((randomValue >> 16) % negLength);
                                irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

                                if (irow < 0 || irow >= vocabSize)
                                    irow = randomValue % (vocabSize - 1) + 1;

                                if (irow == nsStarter)
                                    continue;
                            }
                            auto syn1row = reinterpret_cast<T*>(s1n.getSpecialBuffer()) + (irow * vectorLength);

                            nSampling_<T>(syn0row, syn1row, expTable, neu1e, alpha, vectorLength, r == 0 ? 1 : 0, expLength, false, stream);
                        }
                    }
                    addInfVectorKernel<T><<<128, 256, 256, *stream>>>(syn0row, neu1e, vectorLength);
                    err = cudaStreamSynchronize(*stream);
                    if (0 != err) {
                        throw cuda_exception::build("helpers::skipgramBatchExec_: Cannot synchronize stream after addInfVectorKernel", err);
                    }

                    // optionally release temp arrays
                    err = cudaFree(neu1e);
                    if (err != 0) {
                        throw cuda_exception::build("helpers::skipgramBatchExec_: Cannot deallocate memory with stage", err);
                        break;
                    }
//                    if (vectorLength > 600)
//                        delete[] neu1e;
                }
            }
            BUILD_SINGLE_TEMPLATE(template void skipgramBatchExec_, (NDArray &s0, NDArray &s1, NDArray &s1n, NDArray& expTable, NDArray& negTable, NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr, NDArray &nextRandom, const int nsRounds, const bool preciseMode, const int numThreads), FLOAT_TYPES);

            void skipgram(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable,
                    NDArray &target, NDArray &ngStarter, int nsRounds, NDArray &indices, NDArray &codes, NDArray &alpha, NDArray &randomValue, NDArray &inferenceVector, const bool preciseMode, const int numWorkers) {
                auto xType = syn0.dataType();
                // single round case
                if ((ngStarter.isScalar() && !ngStarter.isEmpty())|| (target.isScalar() && !target.isEmpty())) {
                    auto hsRounds = codes.lengthOf();
                    target.syncToHost();
                    ngStarter.syncToHost();
                    alpha.syncToHost();
                    randomValue.syncToHost();
                    
                    auto targetV = target.isEmpty() ? -1 : target.e<int>(0);
                    auto starterV = ngStarter.isEmpty() ? -1 : ngStarter.e<int>(0);
                    auto alphaV = alpha.e<double>(0);
                    auto randomV = randomValue.e<Nd4jLong>(0);
                    BUILD_SINGLE_SELECTOR(xType, skipgram_, (syn0, syn1, syn1Neg, expTable, negTable, inferenceVector, targetV, starterV, indices, codes, alphaV, randomV, hsRounds, nsRounds), FLOAT_TYPES);
                } else if (ngStarter.isVector() || target.isVector()){
                    // batch mode
//                     NDArray& infVector, NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr, NDArray &nextRandom, const int nsRounds, const bool preciseMode, const int numThreads)
                    BUILD_SINGLE_SELECTOR(xType, skipgramBatchExec_, (syn0, syn1, syn1Neg, expTable, negTable, target, ngStarter, indices, codes, alpha, randomValue, nsRounds, preciseMode, numWorkers), FLOAT_TYPES);
                } else
                    throw std::runtime_error("SkipGram: target must have rank 0 or 1");
            }

            template <typename T>
            static __global__ void checkContextKernel(int* context, T* syn0, T* neu1, int contextWidth, int vectorLength, int vocabSize) {
                __shared__ bool hasError;
                if (0 == threadIdx.x) {
                    hasError = false;
                }
                auto start = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = blockDim.x * gridDim.x;

                for (int c = start; c < contextWidth; c += step) {
                    if (context[c] >= vocabSize)
                        hasError = true; //throw std::runtime_error("Bad context 4");
                    if (!hasError) {
                        T *syn0word = syn0 + (context[c] * vectorLength);

                        for (int i = 0; i < vectorLength; i++) {
                            neu1[i] += syn0word[i];
                        }
                    }
                }
                if (threadIdx.x == 0) {
                    if (hasError)
                        neu1[0] = DataTypeUtils::infOrMax<T>();
                }
                __syncthreads();
            }

            template <typename T>
            __global__ void shiftKernel(T* neu1, T* infVector, int contextWidth, int vectorLength) {
                auto start = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = blockDim.x * gridDim.x;

                for (int i = start; i < vectorLength; i += step) {
                    neu1[i] /= contextWidth + int(infVector != nullptr); // ? 1 : 0);
                }
            }

            template <typename T>
            __global__ void fillUpSynonymsKernel(int starter, int contextWidth, int vectorLength, int* lockedWords, int* context, T* neu1e, T* syn0) {
                auto start = threadIdx.x + blockIdx.x * blockDim.x;
                auto step = blockDim.x * gridDim.x;

                for (int c = starter + start; c < contextWidth; c += step) {
                    if (lockedWords[c] == 1)
                        continue;

                    T *syn0word = syn0 + (context[c] * vectorLength);

                    for (int i = 0; i < vectorLength; i++) {
                        syn0word[i] += neu1e[i];
                    }
                }
            }

            template <typename T>
            void cbow_(LaunchContext* lc, void *vsyn0, void *vsyn1, void *vsyn1Neg, void *vexpTable, void *vnegTable, void *vinfVector, int target, int ngStarter, int *context, int *lockedWords, int *indices, int8_t *codes, double alpha, Nd4jLong randomValue, const int contextWidth, const int hsRounds, const int nsRounds, const int vocabSize, const int vectorLength, const int expLength, const int negLength, const int numLabels, const bool trainWords) {
                auto syn0 = reinterpret_cast<T *>(vsyn0);
                auto syn1 = reinterpret_cast<T *>(vsyn1);
                auto syn1Neg = reinterpret_cast<T *>(vsyn1Neg);
                auto expTable = reinterpret_cast<T *>(vexpTable);
                auto negTable = reinterpret_cast<T *>(vnegTable);
                auto infVector = reinterpret_cast<T *>(vinfVector);
                auto stream = lc->getCudaStream();

                T* neu1; // = new T[vectorLength];
                T* neu1e; // = new T[vectorLength];
                size_t buffSize = sizeof(T) * vectorLength;
                auto err = cudaMalloc(&neu1, buffSize);
                err = cudaMalloc(&neu1e, buffSize);
                err = cudaMemset(neu1, 0, buffSize);
                err = cudaMemset(neu1e, 0, buffSize);

                // building neu1 for current window
                checkContextKernel<T><<<1,1,128,*stream>>>(context, syn0, neu1, contextWidth, vectorLength, vocabSize);

                T checkVal;
                err = cudaMemcpy(&checkVal, neu1, sizeof(T), cudaMemcpyDeviceToHost);
                if (DataTypeUtils::infOrMax<T>() == checkVal)
                    throw std::runtime_error("Bad context 4");
                // for inference we add additional inference vector
                if (infVector != nullptr) {
                    addInfVectorKernel<T><<<128, 256, 128, *stream>>>(neu1, infVector, vectorLength);
                }


                // average neu1
                if (contextWidth > 0) {
                    shiftKernel<T><<<128, 256, 128, *stream>>>(neu1, infVector, contextWidth, vectorLength);
                }

                // softmax round
                if (hsRounds > 0) {
                    for (int i = 0; i < hsRounds; i++) {
                        if (indices[i] < 0 || indices[i] >= vocabSize)
                            throw std::runtime_error("Bad context 5");
                        T* syn1Shifted = syn1 + (indices[i] * vectorLength);
                        hSoftmax_<T>(neu1, syn1Shifted, expTable, neu1e, alpha, vectorLength, codes[i], expLength, infVector != nullptr, stream);
                    }
                }

                auto nsStarter = ngStarter;
                auto irow = nsStarter;
                if (nsRounds > 0) {
                    for (int r = 0; r < nsRounds + 1; r++) {
                        if (r == 0) {
                            // target is known in advance
                        } else {
                            randomValue = randomValue * (unsigned long long) 25214903917 + 11;
                            auto idx = sd::math::nd4j_abs<Nd4jLong >((randomValue >> 16) % negLength);
                            irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

                            if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;
                            if (irow == nsStarter)
                                continue;
                        }

                        nSampling_<T>(neu1, syn1Neg + (irow * vectorLength), expTable, neu1e, alpha, vectorLength, r == 0 ? 1 : 0, expLength, infVector != nullptr, stream);
                    }
                }

                // if we don't train words - we skip start of idxSyn0
                int starter = trainWords == 1 ? 0 : contextWidth - numLabels;

                // propagate neu1e -> syn0
                if (infVector == nullptr) {
                    fillUpSynonymsKernel<T><<<1,1,128, *stream>>>(starter, contextWidth, vectorLength, lockedWords, context, neu1e, syn0);
                } else {

                    for (int i = 0; i < vectorLength; i++) {
                        infVector[i] += neu1e[i];
                    }
                }
                err = cudaStreamSynchronize(*stream);
                if (0 != err) {
                    throw cuda_exception::build(
                            "helpers::cbow_: Cannot synchronize stream after kernel executing", err);
                }
                err = cudaFree(neu1);
                if (0 != err) {
                    throw cuda_exception::build(
                            "helpers::cbow_: Cannot deallocate memory for synonims table", err);
                }

                err = cudaFree(neu1e);
                if (0 != err) {
                    throw cuda_exception::build(
                            "helpers::cbow_: Cannot deallocate memory for antonims table", err);
                }
            }
            BUILD_SINGLE_TEMPLATE(template void cbow_, (LaunchContext* lc, void *syn0, void *syn1, void *syn1Neg, void *expTable, void *vnegTable, void *vinfVector, int target, int ngStarter, int *context, int *lockedWords, int *indices, int8_t *codes, double alpha, Nd4jLong randomValue, const int contextWidth, const int hsRounds, const int nsRounds, const int vocabSize, const int vectorLength, const int expLength, const int negLength, const int numLabels, const bool trainWords), FLOAT_TYPES);

            template <typename T>
            static __global__ void buildCurrentWindowKernel(int vocabSize, int contextWidth, int vectorLength, int* bContext, T* syn0, T* neu1, int* actualContext, int e) {
                // building neu1 for current window
                auto start = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = blockDim.x * gridDim.x;

                for (int c = start; c < contextWidth; c += step) {
                    // getting next context word
                    auto cContext = bContext[c + (e * contextWidth)];

                    // skipping padded values
                    if (cContext < 0)
                        continue;

//                    if (cContext >= vocabSize)
//                        throw std::runtime_error("ContextID can't be >= vocab size");

                    T *syn0word = syn0 + (cContext * vectorLength);

                    for (int i = 0; i < vectorLength; i++)
                        neu1[i] += syn0word[i];

                    atomicAdd(actualContext, 1);
                }
            }

            template <typename T>
            __global__ void arrangeNeuKernel(int vectorLength, T* neu1, T* infVector, int* actualContext) {
                auto start = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = blockDim.x * gridDim.x;

                for (int i = start; i < vectorLength && *actualContext > 0; i += step)
                    neu1[i] /= (*actualContext + int(infVector != nullptr));
            }

            template <typename T>
            __global__ void applyShiftKernel(int* bContext, int* bLocker, T* syn0, T* neu1e, int contextWidth, int vectorLength, int e, int starter) {
                auto step = blockDim.x * gridDim.x;
                auto start = blockDim.x * blockIdx.x + threadIdx.x;

                for (int c = starter + start; c < contextWidth; c += step) {
                    // getting context
                    auto cContext = bContext[c + (e * contextWidth)];
                    auto cLock = bLocker[c + (e * contextWidth)];

                    // skipping padded values
                    if (cContext < 0 || cLock == 1)
                        continue;

//                    if (cContext >= vocabSize)
//                        throw std::runtime_error("ContextID can't be > vocab size");

                    // one word from context
                    T *syn0word = syn0 + (cContext * vectorLength);

                    for (int i = 0; i < vectorLength; i++)
                        syn0word[i] += neu1e[i];

                }
            }

            template <typename T>
            void cbowBatchExec_(LaunchContext* lc, NDArray &s0, NDArray &s1, NDArray &s1n, void *vexpTable, void *vnegTable, void *vinfVector, NDArray &context, NDArray &lockedWords, NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr, NDArray &nextRandom, NDArray &nLabels, const int nsRounds, const int vocabSize, const int vectorLength, const int expLength, const int negLength, const bool trainWords, const int numThreads) {
                const auto syn0 = reinterpret_cast<T*>(s0.specialBuffer()); //bufferAsT<T>();
                const auto syn1 = reinterpret_cast<T*>(s1.specialBuffer()); //bufferAsT<T>();
                const auto syn1Neg = reinterpret_cast<T*>(s1n.specialBuffer()); //bufferAsT<T>();

                const auto expTable = reinterpret_cast<T*>(vexpTable);
                const auto negTable = reinterpret_cast<T*>(vnegTable);
                const auto infVector = reinterpret_cast<T*>(vinfVector);

                auto stream = lc->getCudaStream();

                indices.syncToHost();
                codes.syncToHost();
                negStarters.syncToHost();
                context.syncToHost();

                //const auto numThreads = omp_get_max_threads();
                const auto idxShift = indices.isEmpty() ? 0 : indices.sizeAt(1);
                const auto hsRounds = codes.isEmpty() ? 0 : codes.sizeAt(1);
                const auto numTargets = context.sizeAt(0);
                const int contextWidth = context.sizeAt(1);
                //const auto bContext = reinterpret_cast<int*>(context.buffer()); //bufferAsT<int>();
                const auto dContext = context.dataBuffer()->specialAsT<int>(); //bufferAsT<int>();
//                const auto bLocker = reinterpret_cast<int*>(lockedWords.buffer()); //lockedWords.bufferAsT<int>();
                const auto dLocker = lockedWords.dataBuffer()->specialAsT<int>(); //.specialBuffer()); //lockedWords.bufferAsT<int>();
                const auto bIndices = indices.dataBuffer()->primaryAsT<int>(); //buffer());//AsT<int>();
                const auto bCodes = codes.dataBuffer()->primaryAsT<int8_t>(); //reinterpret_cast<int8_t*>(codes.buffer()); //bufferAsT<int8_t>();
                const auto bStarters = negStarters.dataBuffer()->primaryAsT<int>(); //reinterpret_cast<int*>(negStarters.buffer()); //AsT<int>();
                const auto numIndices = indices.isEmpty() ? 0 : indices.sizeAt(1);
                lr.syncToHost();
                nLabels.syncToHost();
                //PRAGMA_OMP_PARALLEL_FOR_ARGS(num_threads(numThreads) private(sneu1, sneu1e))
                //NDArray neuVector('c', {vectorLength}, DataTypeUtils::fromT<T>());
               // auto neuEVector = neuVector; //NDArrayFactory::create<T>('c', {vectorLength});
                T* neu1; // = reinterpret_cast<T*>(neuVector.specialBuffer());// = vectorLength <= 600 ? sneu1 : new T[vectorLength];
                T* neu1e; // = reinterpret_cast<T*>(neuVector.specialBuffer()); // = vectorLength <= 600 ? sneu1e : new T[vectorLength];
                auto cerr = cudaMalloc(&neu1, sizeof(T) * vectorLength);
                if (cerr) {
                    throw cuda_exception::build("Cannot allocate temp vector buffer", cerr);
                }
                cerr = cudaMalloc(&neu1e, sizeof(T) * vectorLength);
                if (cerr) {
                    throw cuda_exception::build("Cannot allocate temp vector buffer", cerr);
                }
                int* actualContext;
                cerr = cudaMalloc(&actualContext, sizeof(int));
                if (cerr) {
                    throw cuda_exception::build("Cannot allocate counter buffer", cerr);
                }

                for (int e = 0; e < numTargets; e++) {

//                    auto err = cudaMalloc(&neu1, sizeof(T)* vectorLength);
//                   q err = cudaMalloc(&neu1e, sizeof(T)*vectorLength);
//
//                    // optionally we nullify temp arrays after successful (and on first) cycle
//                    memset(neu1, 0, sizeof(T) * vectorLength);
//                    memset(neu1e, 0, sizeof(T) * vectorLength);

                    auto alpha = lr.e<double>(e);
                    auto numLabels = nLabels.isEmpty() ? 0 : nLabels.e<int>(e);

//                    auto err = cudaMemset(actualContext, 0, sizeof(int));
//                    if (err) {
//                        printf("Cuda error %d\n", err); break;
//                    }

                    buildCurrentWindowKernel<T><<<1,1,128, *stream>>>(vocabSize, contextWidth, vectorLength, dContext, syn0, neu1, actualContext, e);
                    arrangeNeuKernel<T><<<1,1,128, *stream>>>(vectorLength, neu1, infVector, actualContext);

                    // hierarchic softmax step
                    if (!indices.isEmpty()) {
                        for (int i = 0; i < numIndices; i++) {
                            const int cIndex = bIndices[(e * numIndices) + i];
                            const int cCode = bCodes[(e * numIndices) + i];

                            // we're skipping padded values
                            if (cIndex < 0)
                                continue;

                            if (cIndex >= vocabSize)
                                throw std::runtime_error("Index can't be > vocab size");

                            hSoftmax_<T>(neu1, syn1 + (cIndex * vectorLength), expTable, neu1e, alpha, vectorLength, cCode, expLength, false, stream);
                        }
                    }

                    // negative sampling step
                    if (!negStarters.isEmpty() && nsRounds > 0) {
                        int irow = bStarters[e];
                        const int nsStarter = irow;
                        unsigned long long randomValue = nextRandom.e<Nd4jLong>(e);

                        for (int r = 0; r < nsRounds + 1; r++) {
                            // we're skipping rng on 0 step
                            if (r != 0) {
                                randomValue = randomValue * (unsigned long long) 25214903917 + 11;
                                auto idx = sd::math::nd4j_abs<Nd4jLong>((randomValue >> 16) % negLength);
                                irow = idx >= negLength ? -1 : static_cast<int>(negTable[idx]);

                                if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;
                                if (irow == nsStarter)
                                    continue;

                                nSampling_<T>(neu1, s1n.bufferWithOffset(irow * vectorLength), expTable, neu1e, alpha, vectorLength, r == 0 ? 1 : 0, expLength, infVector != nullptr, stream);
                            } else {
                                nSampling_<T>(neu1, s1n.bufferWithOffset(irow * vectorLength), expTable, neu1e, alpha, vectorLength, r == 0 ? 1 : 0, expLength, infVector != nullptr, stream);
                            }

                            //nd4j_printf("Thread <%i>: syn0: [%i]; s1n: [%i];\n", omp_get_thread_num(), 0, irow);
                        }
                    }


                    // if we're skipping labels
                    int starter = trainWords == 1 ? 0 : contextWidth - numLabels;

                    // applying previously averaged results
                    applyShiftKernel<T><<<1,1,128, *stream>>>(dContext, dLocker, syn0, neu1e, contextWidth, vectorLength, e, starter);

                    // optionally release temp arrays
//                    if (vectorLength > 600) {
//                    }

                }
                cerr = cudaStreamSynchronize(*stream);
                if (cerr) {
                    throw cuda_exception::build("Cannot syncronize stream before memory deallocation", cerr);
                }

                cerr = cudaFree(neu1);
                if (cerr) {
                    throw cuda_exception::build("Cannot deallocate temp buffer1", cerr);
                }
                cerr = cudaFree(neu1e);
                if (cerr) {
                    throw cuda_exception::build("Cannot deallocate temp buffer1 E", cerr);
                }
                cerr = cudaFree(actualContext);
                if (cerr) {
                    throw cuda_exception::build("Cannot deallocate temp buffer1", cerr);
                }

            }
            BUILD_SINGLE_TEMPLATE(template void cbowBatchExec_, (LaunchContext* lc, NDArray &s0, NDArray &s1, NDArray &s1n, void *vexpTable, void *vnegTable, void *vinfVector, NDArray &context, NDArray &lockedWords, NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr, NDArray &nextRandom, NDArray &nLabels, const int nsRounds, const int vocabSize, const int vectorLength, const int expLength, const int negLength,  const bool trainWords, const int numThreads), FLOAT_TYPES);

            void cbow(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, NDArray &target, NDArray &ngStarter, int nsRounds, NDArray &context, NDArray &lockedWords, NDArray &indices, NDArray &codes, NDArray &alpha, NDArray &randomValue, NDArray &numLabels, NDArray &inferenceVector, const bool trainWords, int numWorkers) {
                auto xType = syn0.dataType();
                auto lc = context.getContext();
                indices.syncToHost();
                NDArray::prepareSpecialUse({&syn0, &syn1, &syn1Neg, &expTable, &negTable, &target, &ngStarter}, {&context, &lockedWords, &indices, &codes, &alpha, &randomValue, &numLabels, &inferenceVector});
                //auto stream = lc->getCudaStream();
                if ((context.rankOf() == 0 || context.rankOf() == 1) && (indices.rankOf() == 1 || indices.rankOf() == 0)) {
                    // single round case
                    /*nd4j_printf("Row exec; ContextWidth: %i; LockedWords: %i; numLabels: %i; Train words: %i\n", (int) context.lengthOf(), (int) lockedWords.lengthOf(), numLabels.isEmpty() ? 0 : numLabels.e<int>(0), (int) trainWords);
                    if (context.lengthOf() == 2) {
                        context.printBuffer("context");
                        lockedWords.printBuffer("locked");
                        codes.printBuffer("codes");
                        indices.printBuffer("indices");
                    }*/

                    auto hsRounds = codes.lengthOf();
                    target.syncToHost();
                    numLabels.syncToHost();
                    target.syncToHost();
                    alpha.syncToHost();
                    numLabels.syncToHost();
                    codes.syncToHost();
                    negTable.syncToHost();
                    BUILD_SINGLE_SELECTOR(xType, cbow_, (lc, syn0.specialBuffer(), syn1.specialBuffer(), syn1Neg.specialBuffer(), expTable.specialBuffer(), negTable.buffer(), inferenceVector.specialBuffer(), target.isEmpty() ? -1 : target.e<int>(0), ngStarter.isEmpty() ? -1 : ngStarter.e<int>(0), reinterpret_cast<int *>(context.specialBuffer()), reinterpret_cast<int *>(lockedWords.specialBuffer()),reinterpret_cast<int *>(indices.buffer()), reinterpret_cast<int8_t *>(codes.buffer()), alpha.e<double>( 0), randomValue.e<Nd4jLong>(0), (int) context.lengthOf(), hsRounds, nsRounds, (int) syn0.sizeAt(0), (int) syn0.sizeAt(1), (int) expTable.lengthOf(), (int) negTable.lengthOf(), numLabels.isEmpty() ? 0 : numLabels.e<int>(0), trainWords), FLOAT_TYPES);
                } else if (context.rankOf() == 2 && indices.rankOf() == 2) {
                    // batch mode
                    //nd4j_printf("Batch exec\n","");

                    BUILD_SINGLE_SELECTOR(xType, cbowBatchExec_, (lc, syn0, syn1, syn1Neg, expTable.specialBuffer(), negTable.specialBuffer(), nullptr, context, lockedWords, target, ngStarter, indices, codes, alpha, randomValue, numLabels, nsRounds, syn0.sizeAt(0), syn0.sizeAt(1), expTable.lengthOf(), negTable.isEmpty() ? 0 : negTable.lengthOf(), trainWords, numWorkers), FLOAT_TYPES);
                } else
                    throw std::runtime_error("CBOW: context must have rank 0/1 or 2");

                NDArray::registerSpecialUse({&syn0, &syn1, &syn1Neg, &expTable, &negTable, &target, &ngStarter}, {&context, &lockedWords, &indices, &codes, &alpha, &randomValue, &numLabels, &inferenceVector});
            }

        }
    }
}