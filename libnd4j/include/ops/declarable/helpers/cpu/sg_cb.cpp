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
#include <AveragingArrayProxy.h>
#include <helpers/AveragingArrayProxy.h>
#include <specials.h>

#define HS_MAX_EXP 6.0f

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename T>
            void hSoftmax_(void *vsyn0, void *vsyn1, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code, int expLength, bool isInference) {
                auto syn0 = reinterpret_cast<T*>(vsyn0);
                auto syn1 = reinterpret_cast<T*>(vsyn1);
                auto expTable = reinterpret_cast<T*>(vexpTable);
                auto neu1e = reinterpret_cast<T*>(vneu1e);

                T dot(0.0f);
                T g(0.0f);
                T f(0.0f);

                // dot
#pragma omp simd reduction(sumT:dot)
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
#pragma omp simd
                for (int e = 0; e < vectorLength; e++) {
                    neu1e[e] = g * syn1[e] + neu1e[e];
                }

                // axpy2
                if (!isInference) {
#pragma omp simd
                    for (int e = 0; e < vectorLength; e++) {
                        syn1[e] = g * syn0[e] + syn1[e];
                    }
                }
            }

            template <typename T>
            void nSampling_(void *vsyn0, void *vsyn1Neg, void *vexpTable, void *vneu1e, double alpha, int vectorLength, int code, int expLength, bool isInference) {
                auto syn0 = reinterpret_cast<T*>(vsyn0);
                auto syn1Neg = reinterpret_cast<T*>(vsyn1Neg);
                auto expTable = reinterpret_cast<T*>(vexpTable);
                auto neu1e = reinterpret_cast<T*>(vneu1e);

                T dot = (T) 0.0f;
                T g = (T) 0.0f;

                #pragma omp simd reduction(sumT:dot)
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
                #pragma omp simd
                for (int e = 0; e < vectorLength; e++) {
                    neu1e[e] = g * syn1Neg[e] + neu1e[e];
                }

                // axpy2
                if (!isInference) {

                    #pragma omp simd
                    for (int e = 0; e < vectorLength; e++) {
                        syn1Neg[e] = g * syn0[e] + syn1Neg[e];
                    }
                }
            }

            template <typename T>
            void cbow_(void *vsyn0, void *vsyn1, void *vsyn1Neg, void *vexpTable, void *vnegTable, void *vinfVector, int target, int ngStarter, int *context, int *indices, int8_t *codes, double alpha, Nd4jLong randomValue, const int contextWidth, const int hsRounds, const int nsRounds, const int vocabSize, const int vectorLength, const int expLength, const int negLength, const int numLabels, const bool trainWords) {
                auto syn0 = reinterpret_cast<T *>(vsyn0);
                auto syn1 = reinterpret_cast<T *>(vsyn1);
                auto syn1Neg = reinterpret_cast<T *>(vsyn1Neg);
                auto expTable = reinterpret_cast<T *>(vexpTable);
                auto negTable = reinterpret_cast<int *>(vnegTable);
                auto infVector = reinterpret_cast<T *>(vinfVector);

                auto neu1 = new T[vectorLength];
                auto neu1e = new T[vectorLength];
                memset(neu1, 0, vectorLength * sizeof(T));
                memset(neu1e, 0, vectorLength * sizeof(T));

                // building neu1 for current window
                for (int c = 0; c < contextWidth; c++) {
                    T *syn0word = syn0 + (context[c] * vectorLength);

                    #pragma omp simd
                    for (int i = 0; i < vectorLength; i++) {
                        neu1[i] += syn0word[i];
                    }
                }

                // for inference we add additional inference vector
                if (infVector != nullptr) {

                    #pragma omp simd
                    for (int i = 0; i < vectorLength; i++) {
                        neu1[i] += infVector[i];
                    }
                }


                // average neu1
                if (contextWidth > 0) {

                    #pragma omp simd
                    for (int i = 0; i < vectorLength; i++) {
                        neu1[i] /= contextWidth + (infVector != nullptr ? 1 : 0);
                    }
                }

                // softmax round
                if (hsRounds > 0) {
                    for (int i = 0; i < hsRounds; i++) {
                        hSoftmax_<T>(neu1, syn1 + (indices[i] * vectorLength), expTable, neu1e, alpha, vectorLength, codes[i], expLength, infVector != nullptr);
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
                            auto idx = nd4j::math::nd4j_abs<Nd4jLong >((randomValue >> 16) % negLength);
                            irow = idx >= negLength ? -1 : negTable[idx];

                            if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;
                            if (irow == nsStarter)
                                continue;
                        }

                        nSampling_<T>(neu1, syn1Neg + (irow * vectorLength), expTable, neu1e, alpha, vectorLength, r == 0 ? 1 : 0, expLength, infVector != nullptr);
                    }
                }

                // if we don't train words - we skip start of idxSyn0
                int starter = trainWords == 1 ? 0 : contextWidth - numLabels;

                // propagate neu1e -> syn0
                if (infVector == nullptr) {
                    for (int c = starter; c < contextWidth; c++) {
                        T *syn0word = syn0 + (context[c] * vectorLength);

                        #pragma omp simd
                        for (int i = 0; i < vectorLength; i++) {
                            syn0word[i] += neu1e[i];
                        }
                    }
                } else {

                    #pragma omp simd
                    for (int i = 0; i < vectorLength; i++) {
                        infVector[i] += neu1e[i];
                    }
                }


                delete[] neu1;
                delete[] neu1e;
            }
            BUILD_SINGLE_TEMPLATE(template void cbow_, (void *syn0, void *syn1, void *syn1Neg, void *expTable, void *vnegTable, void *vinfVector, int target, int ngStarter, int *context, int *indices, int8_t *codes, double alpha, Nd4jLong randomValue, const int contextWidth, const int hsRounds, const int nsRounds, const int vocabSize, const int vectorLength, const int expLength, const int negLength, const int numLabels, const bool trainWords), FLOAT_TYPES);


            template <typename T>
            void skipgram_(void *vsyn0, void *vsyn1, void *vsyn1Neg, void *vexpTable, void *vnegTable, void *vinfVector, int target, int ngStarter, int *indices, int8_t *codes, double alpha, Nd4jLong randomValue, const int hsRounds, const int nsRounds, const int vocabSize, const int vectorLength, const int expLength, const int negLength) {
                auto syn0 = reinterpret_cast<T*>(vsyn0);
                auto syn1 = reinterpret_cast<T*>(vsyn1);
                auto syn1Neg = reinterpret_cast<T*>(vsyn1Neg);
                auto expTable = reinterpret_cast<T*>(vexpTable);
                auto negTable = reinterpret_cast<int*>(vnegTable);
                auto infVector = reinterpret_cast<T*>(vinfVector);

                auto neu1e = new T[vectorLength];
                memset(neu1e, 0, vectorLength * sizeof(T));

                // hierarchic softmax goes first (if enabled)
                auto syn0row = syn0 + (target * vectorLength);
                auto irow = 0;
                if (hsRounds > 0) {
                    for (int r = 0; r < hsRounds; r++) {
                        irow = indices[r];
                        if (irow < 0 || irow >= vocabSize)
                            break;

                        hSoftmax_<T>(syn0row, syn1 + (irow * vectorLength), expTable, neu1e, alpha, vectorLength, codes[r], expLength, infVector != nullptr);
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
                            auto idx = nd4j::math::nd4j_abs<Nd4jLong >((randomValue >> 16) % negLength);
                            irow = idx >= negLength ? -1 : negTable[idx];

                            if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;
                            if (irow == nsStarter)
                                continue;
                        }

                       nSampling_<T>(syn0row, syn1Neg + (irow * vectorLength), expTable, neu1e, alpha, vectorLength, r == 0 ? 1 : 0, expLength, infVector != nullptr);
                    }
                }

                if (infVector == nullptr) {
#pragma omp simd
                    for (int e = 0; e < vectorLength; e++) {
                        syn0row[e] += neu1e[e];
                    }
                } else {
#pragma omp simd
                    for (int e = 0; e < vectorLength; e++) {
                        infVector[e] += neu1e[e];
                    }
                }

                delete[] neu1e;
            }
            BUILD_SINGLE_TEMPLATE(template void skipgram_, (void *syn0, void *syn1, void *syn1Neg, void *expTable, void *vnegTable, void *vinfVector, int target, int ngStarter, int *indices, int8_t *codes, double alpha, Nd4jLong randomValue, const int hsRounds, const int nsRounds, const int vocabSize, const int vectorLength, const int expLength, const int negLength), FLOAT_TYPES);

            bool search_(int *haystack, int needle, int totalElements) {
                int firstIndex = 0;
                int lastIndex = totalElements - 1;
                int halfIndex = nd4j::math::nd4j_floor<float, int>((lastIndex + firstIndex) / (float) 2);

                while(haystack[halfIndex] != needle && firstIndex < lastIndex) {
                    if (needle < haystack[halfIndex]) {
                        lastIndex = halfIndex - 1;
                    } else if (needle > haystack[halfIndex]) {
                        firstIndex = halfIndex + 1;
                    }
                    halfIndex = nd4j::math::nd4j_floor<float, int>((lastIndex + firstIndex) / (float) 2);
                }

                return (haystack[halfIndex] != needle) ? false : true;
            }

            template <typename T>
            void skipgramBatchExec_(NDArray &s0, NDArray &s1, NDArray &s1n, void *vexpTable, void *vnegTable, void *vinfVector, NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr, NDArray &nextRandom, const int nsRounds, const int vocabSize, const int vectorLength, const int expLength, const int negLength) {
                //auto syn0 = reinterpret_cast<T*>(vsyn0);
                //auto syn1 = reinterpret_cast<T*>(vsyn1);
                //auto syn1Neg = reinterpret_cast<T*>(vsyn1Neg);
                const auto expTable = reinterpret_cast<T*>(vexpTable);
                const auto negTable = reinterpret_cast<int*>(vnegTable);
                const auto infVector = reinterpret_cast<T*>(vinfVector);

                T sneu1e[600];

                const auto numThreads = 6;
                const auto idxShift = indices.isEmpty() ? 0 : indices.sizeAt(1);
                const auto hsRounds = codes.isEmpty() ? 0 : codes.sizeAt(1);



                if (!indices.isEmpty()) {
                    auto bTarget = targets.bufferAsT<int>();
                    auto bIndices = indices.bufferAsT<int>();
                    auto bCodes = codes.bufferAsT<int8_t>();
                    auto numTargets = targets.lengthOf();


// parallel block and following loop will be the same for every thread
#pragma omp parallel num_threads(numThreads)  private(sneu1e) default(shared)
                    {
                        auto isOwner = true;
                        auto irow = 0;

                        // if vectorLength > pre-defined value we'll allocate new array
                        T* neu1e = vectorLength <= 600 ? sneu1e : new T[vectorLength];

                        // initial target position
                        // f can't be higher than batch size
                        int f = omp_get_thread_num() > numTargets ? omp_get_thread_num() % numTargets : omp_get_thread_num();

                        for (int t = 0; t < numTargets; t++) {
                            // this value should be different for all threads, so we're shifting values here
                            if (f >= numTargets)
                                f = 0;

                            // actual target for THIS thread
                            auto target = bTarget[f];
                            auto alpha = lr.e<double>(f);

                            // if previous cycle used neu1e - nullify it
                            if (isOwner)
                                memset(neu1e, 0, vectorLength * sizeof(T));

                            // we're deciding if this thread will process this given target, or not
                            isOwner = target % numThreads == omp_get_thread_num();
                            auto syn0row = isOwner ? reinterpret_cast<T*>(s0.bufferWithOffset(target * vectorLength)) : 0;

                            auto cShift = f * idxShift;

                            int x = omp_get_thread_num() > hsRounds ? omp_get_thread_num() % hsRounds : omp_get_thread_num();

                            for (int r = 0; r < hsRounds; r++) {
                                // this row should be randomized as well, to reduce chances for race conditions
                                if (x >= hsRounds)
                                    x = 0;

                                bool isSkipRound = false;

                                irow = bIndices[x + cShift];
                                if (irow < 0 || irow >= vocabSize) {
                                    isSkipRound = true;
                                }

                                // all threads diverge here on top of divergence over syn0 table
                                if (isOwner && !isSkipRound) {
                                    auto syn1row = s1.bufferWithOffset(irow * vectorLength);
                                    auto code = bCodes[x + cShift];

                                    //nd4j_printf("syn0: [%i]; syn1: [%i]; code: [%i]\n", target, irow, code);
                                    hSoftmax_<T>(syn0row, syn1row, expTable, neu1e, alpha, vectorLength, code, expLength, infVector != nullptr);
                                }

                                x++;
                            }

                            if (isOwner) {
                                for (int e = 0; e < vectorLength; e++) {
                                    syn0row[e] += neu1e[e];
                                }
                            }

                            // we synchronize all threads here so they move synchronously
                            //#pragma omp barrier

                            // now we increment further step
                            f++;
                        }

                        // optional deallocation
                        if (vectorLength > 600)
                            delete[] neu1e;
                    }
                }

                // negative sampling goes second (if enabled)
                if (nsRounds > 0) {
                    const auto numTargets = targets.lengthOf();
                    const auto bTarget = targets.bufferAsT<int>();
                    const auto bStarters = negStarters.bufferAsT<int>();

                    //copy & sort starters
                    auto sStarters = new int[numTargets];
                    memcpy(sStarters, bStarters, numTargets * sizeof(int));
                    SpecialMethods<int>::sortGeneric(sStarters, negStarters.shapeInfo(), false);

// same parallelism here, group by target AND nsStarter pair
#pragma omp parallel num_threads(numThreads) private(sneu1e) default(shared)
                    {

                        auto isOwner = true;
                        auto irow = 0;

                        // if vectorLength > pre-defined value we'll allocate new array
                        T* neu1e = vectorLength <= 600 ? sneu1e : new T[vectorLength];

                        // initial target position
                        // f can't be higher than batch size
                        int f = omp_get_thread_num() > numTargets ? omp_get_thread_num() % numTargets : omp_get_thread_num();

                        for(int t = 0; t < numTargets; t++) {
                            // this value should be different for all threads, so we're shifting values here
                            if (f >= numTargets)
                                f = 0;

                            // actual target for THIS thread
                            auto target = bTarget[f];
                            auto nsStarter = bStarters[f];
                            auto alpha = lr.e<double>(f);
                            auto randomValue = nextRandom.e<Nd4jLong>(f);

                            // if previous cycle used neu1e - nullify it
                            if (isOwner)
                                memset(neu1e, 0, vectorLength * sizeof(T));

                            // we're deciding if this thread will process this given target, or not
                            isOwner = (((target * 31) + nsStarter) * 31) % numThreads == omp_get_thread_num();
                            auto syn0row = isOwner ? reinterpret_cast<T*>(s0.bufferWithOffset(target * vectorLength)) : 0;

                            irow = nsStarter;
                            if (isOwner) {
                                for (int r = 0; r < nsRounds + 1; r++) {
                                    // we're skipping rng on 0 step
                                    if (r != 0) {
                                        randomValue = nd4j::math::nd4j_abs<Nd4jLong>(randomValue * (unsigned long long) 25214903917 + 11);
                                        auto idx = nd4j::math::nd4j_abs<Nd4jLong>((randomValue >> 16) % negLength);
                                        irow = idx >= negLength ? -1 : negTable[idx];

                                        if (irow < 0 || irow >= vocabSize) irow = randomValue % (vocabSize - 1) + 1;
                                        if (irow == nsStarter)
                                            continue;

                                        // we shift irow here to guarantee independence
                                        int dim = irow % numThreads;
                                        if (dim != omp_get_thread_num()) {
                                            irow += numThreads - omp_get_thread_num();

                                            //if (irow % numThreads != omp_get_thread_num())
                                            //    throw std::runtime_error("boom");

                                            // roll back to nearest affilated word
                                            if (irow >= vocabSize)
                                                irow -= numThreads;

                                            /*
                                            if (search_(sStarters, irow, numTargets)) {
                                                if (irow < numTargets - numThreads)
                                                    irow += numThreads;
                                                else if (irow > numThreads)
                                                    irow -= numThreads;
                                            }
                                            */
                                        }
                                    }

                                    //nd4j_printf("Thread <%i>: syn0: [%i]; s1n: [%i];\n", omp_get_thread_num(), target, irow);
                                    nSampling_<T>(syn0row, s1n.bufferWithOffset(irow * vectorLength), expTable, neu1e, alpha, vectorLength, r == 0 ? 1 : 0, expLength, infVector != nullptr);
                                }


                                for (int e = 0; e < vectorLength; e++) {
                                    syn0row[e] += neu1e[e];
                                }
                            }

                            f++;
                        }

                        // optional deallocation
                        if (vectorLength > 600)
                            delete[] neu1e;
                    }

                    // deleting sorted stuff
                    delete[] sStarters;
                }
            }
            BUILD_SINGLE_TEMPLATE(template void skipgramBatchExec_, (NDArray &s0, NDArray &s1, NDArray &s1n, void *vexpTable, void *vnegTable, void *vinfVector, NDArray &targets, NDArray &negStarters, NDArray &indices, NDArray &codes, NDArray &lr, NDArray &nextRandom, const int nsRounds, const int vocabSize, const int vectorLength, const int expLength, const int negLength), FLOAT_TYPES);

            void skipgram(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, NDArray &target, NDArray &ngStarter, int nsRounds, NDArray &indices, NDArray &codes, NDArray &alpha, NDArray &randomValue, NDArray &inferenceVector) {
                auto xType = syn0.dataType();

                // single round hase
                if ((ngStarter.isScalar() && !ngStarter.isEmpty())|| (target.isScalar() && !target.isEmpty())) {
                    auto hsRounds = codes.lengthOf();

                    BUILD_SINGLE_SELECTOR(xType, skipgram_, (syn0.buffer(), syn1.buffer(), syn1Neg.buffer(), expTable.buffer(), negTable.buffer(), inferenceVector.buffer(), target.isEmpty() ? -1 : target.e<int>(0), ngStarter.isEmpty() ? -1 : ngStarter.e<int>(0), reinterpret_cast<int *>(indices.buffer()), reinterpret_cast<int8_t *>(codes.buffer()), alpha.e<double>(0), randomValue.e<Nd4jLong>(0), hsRounds, nsRounds, (int) syn0.sizeAt(0), (int) syn0.sizeAt(1), (int) expTable.lengthOf(), (int) negTable.lengthOf()), FLOAT_TYPES);
                } else if (ngStarter.isVector() || target.isVector()){
                    // batch mode

                    BUILD_SINGLE_SELECTOR(xType, skipgramBatchExec_, (syn0, syn1, syn1Neg, expTable.buffer(), negTable.buffer(), nullptr, target, ngStarter, indices, codes, alpha, randomValue, nsRounds, syn0.sizeAt(0), syn0.sizeAt(1), expTable.lengthOf(), negTable.lengthOf()), FLOAT_TYPES);
                } else
                    throw std::runtime_error("SkipGram: Codes must have rank 1 or 2");
            }

            void cbow(NDArray &syn0, NDArray &syn1, NDArray &syn1Neg, NDArray &expTable, NDArray &negTable, NDArray &target, NDArray &ngStarter, int nsRounds, NDArray &context, NDArray &indices, NDArray &codes, NDArray &alpha, NDArray &randomValue, NDArray &inferenceVector, const int numLabels, const bool trainWords) {
                auto xType = syn0.dataType();

                auto hsRounds = codes.lengthOf();

                BUILD_SINGLE_SELECTOR(xType, cbow_, (syn0.buffer(), syn1.buffer(), syn1Neg.buffer(), expTable.buffer(), negTable.buffer(), inferenceVector.buffer(), target.isEmpty() ? -1 : target.e<int>(0), ngStarter.isEmpty() ? -1 : ngStarter.e<int>(0), reinterpret_cast<int*>(context.buffer()), reinterpret_cast<int *>(indices.buffer()), reinterpret_cast<int8_t*>(codes.buffer()), alpha.e<double>(0), randomValue.e<Nd4jLong>(0), (int) context.lengthOf(), hsRounds, nsRounds, (int) syn0.sizeAt(0), (int) syn0.sizeAt(1), (int) expTable.lengthOf(), (int) negTable.lengthOf(), numLabels, trainWords), FLOAT_TYPES);
            }
        }
    }
}