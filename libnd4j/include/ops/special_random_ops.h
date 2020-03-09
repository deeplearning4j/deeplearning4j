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

#ifndef LIBND4J_SPECIAL_RANDOM_OPS_H
#define LIBND4J_SPECIAL_RANDOM_OPS_H

#include <ops/random_ops.h>
#include <helpers/shape.h>
#include <graph/RandomGenerator.h>
#include <ops/specials_cuda.h>
#include <execution/Threads.h>

namespace randomOps {

//////////////////////////////////////////////////////////////////////
    template<typename T>
    class Choice {
    public:

        method_idx
        method_X
        method_XY

        static const bool requiresSpecial = true;


#ifdef __CUDACC__
        __device__ static inline void specialOpCuda(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            /**
             * X holds data,
             * Y holds probabilities
             * Z will hold results
             */

            // TODO: we probably might want to skip this sum, and state that probabilities array should be real probabilities, i.e. should sum to 1.0
            //T probSum = extraArguments[0];

            __shared__ Nd4jLong xLength;
            __shared__ Nd4jLong yLength;
            __shared__ Nd4jLong zLength;

            __shared__ Nd4jLong xEWS;
            __shared__ Nd4jLong yEWS;
            __shared__ Nd4jLong zEWS;
            __shared__ char xOrder;
            __shared__ char yOrder;
            __shared__ char zOrder;

            __shared__ sd::graph::RandomGenerator *rng;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ sd::graph::RandomGenerator *devRng;

            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                rng = (sd::graph::RandomGenerator*) shmem;
                cB = shmem;
                devRng = reinterpret_cast<sd::graph::RandomGenerator*> (state);
                dB = reinterpret_cast<unsigned char *> (state);

                xLength = shape::length(xShapeBuffer);
                yLength = shape::length(yShapeBuffer);
                zLength = shape::length(zShapeBuffer);

                xEWS = shape::elementWiseStride(xShapeBuffer);
                yEWS = shape::elementWiseStride(yShapeBuffer);
                zEWS = shape::elementWiseStride(zShapeBuffer);
                xOrder = shape::order(xShapeBuffer);
                yOrder = shape::order(yShapeBuffer);
                zOrder = shape::order(zShapeBuffer);
            }
            __syncthreads();

            // using this loop instead of memcpy
            for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e+= blockDim.x)
                cB[e] = dB[e];

            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (zEWS >= 1 && xEWS >= 1 && yEWS >= 1 && xOrder == yOrder && xOrder == zOrder) {
                for (Nd4jLong e = tid; e < zLength; e+=blockDim.x * gridDim.x) {
                    T prob = rng->relativeT<T>(e);
                    T cumProb = (T) 0.0f;
                    for (Nd4jLong f = 0; f < yLength; f++) {
                        T relProb = y[f * yEWS];
                        cumProb += relProb;

                        if (prob <= cumProb || f == yLength - 1) {
                            z[e * zEWS] = x[f * xEWS];
                            f += yLength;
                        }
//                        __syncthreads();  // Eliminated due RTX20xx specific
                    }
//                    __syncthreads();  // Eliminated due RTX20xx specific
                }
            }
            else {

                for (Nd4jLong i = tid; i < zLength; i+=blockDim.x * gridDim.x) {

                    auto zOffset2 = shape::getIndexOffset(i, zShapeBuffer);
                    T prob = rng->relativeT<T>(i);
                    T cumProb = (T) 0.0f;

                    for (Nd4jLong f = 0; f < yLength; f++) {

                        auto yOffset2 = shape::getIndexOffset(f, yShapeBuffer);
                        T relProb = y[yOffset2];
                        cumProb += relProb;

                        if (prob <= cumProb || f == yLength - 1) {

                            auto xOffset2 = shape::getIndexOffset(f, xShapeBuffer);
                            z[zOffset2] = x[xOffset2];
                            f += yLength;
                        }
//                        __syncthreads();  // Eliminated due RTX20xx specific
                    }
//                    __syncthreads();  // Eliminated due RTX20xx specific
                }
            }
        }
#endif

        static inline void specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            /**
             * X holds data,
             * Y holds probabilities
             * Z will hold results
             */

            //sd::random::RandomBuffer *buffer = reinterpret_cast<sd::random::RandomBuffer *> (state);
            sd::graph::RandomGenerator* rng = reinterpret_cast<sd::graph::RandomGenerator*>(state);
            // TODO: we probably might want to skip this sum, and state that probabilities array should be real probabilities, i.e. should sum to 1.0
            //T probSum = extraArguments[0];

            auto xLength = shape::length(xShapeBuffer);
            auto yLength = shape::length(yShapeBuffer);
            auto zLength = shape::length(zShapeBuffer);

            auto xEWS = shape::elementWiseStride(xShapeBuffer);
            auto yEWS = shape::elementWiseStride(yShapeBuffer);
            auto zEWS = shape::elementWiseStride(zShapeBuffer);

            int elementsPerThread = zLength / TAD_THRESHOLD;
            int _threads = sd::math::nd4j_max<int>(1, elementsPerThread);
            _threads = sd::math::nd4j_min<int>(_threads, sd::Environment::getInstance()->maxThreads());

            if (zEWS >= 1 && xEWS >= 1 && yEWS >= 1) {
                auto func = PRAGMA_THREADS_FOR {
                    for (auto e = start; e < stop; e++) {
                        T prob = rng->relativeT<T>(e);
                        T cumProb = (T) 0.0f;
                        for (Nd4jLong f = 0; f < yLength; f++) {
                            T relProb = y[f * yEWS];
                            cumProb += relProb;

                            if (prob <= cumProb || f == yLength - 1) {
                                z[e * zEWS] = x[f * xEWS];
                                break;
                            }
                        }
                    }
                };

                sd::Threads::parallel_for(func, 0, zLength, 1, _threads);
            }
            else {

                auto func = PRAGMA_THREADS_FOR {
                    for (Nd4jLong i = 0; i < zLength; i++) {

                        auto zOffset2 = shape::getIndexOffset(i, zShapeBuffer);
                        T prob = rng->relativeT<T>(i);
                        T cumProb = (T) 0.0f;

                        for (Nd4jLong f = 0; f < yLength; f++) {

                            auto yOffset2 = shape::getIndexOffset(f, yShapeBuffer);
                            T relProb = y[yOffset2];
                            cumProb += relProb;

                            if (prob <= cumProb || f == yLength - 1) {

                                auto xOffset2 = shape::getIndexOffset(f, xShapeBuffer);
                                z[zOffset2] = x[xOffset2];
                                break;
                            }
                        }
                    }
                };

                sd::Threads::parallel_for(func, 0, zLength, 1, _threads);
            }
        }
    };


//////////////////////////////////////////////////////////////////////
    /**
    * This Op produces random values within specified boundaries. Distribuion is Gaussian
    */
    template<typename T>
    class GaussianDistribution {
    public:

        method_XY
        method_X
        method_idx

        static const bool requiresSpecial = true;

#ifdef __CUDACC__
        __device__ static inline void specialOpCuda(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {

            __shared__ T epsilon;
            __shared__ T two_pi;

            __shared__ Nd4jLong zLength;
            __shared__ Nd4jLong zEWS;
            __shared__ Nd4jLong yEWS;
            __shared__ T mean;
            __shared__ T stddev;
            __shared__ int step;

            __shared__ T *tZ;

            __shared__ sd::graph::RandomGenerator* rng;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ sd::graph::RandomGenerator *devRng;

            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                rng = reinterpret_cast<sd::graph::RandomGenerator*>(shmem);
                cB = shmem;
                devRng = reinterpret_cast<sd::graph::RandomGenerator *> (state);
                dB = reinterpret_cast<unsigned char *> (state);

                tZ = reinterpret_cast<T *>(shmem + sizeof(sd::graph::RandomGenerator));

                zLength = shape::length(zShapeBuffer);
                zEWS = shape::elementWiseStride(zShapeBuffer);
                yEWS = shape::elementWiseStride(yShapeBuffer);


                epsilon = static_cast<T>(1e-5);
                two_pi = static_cast<T>(2.0f) * static_cast<T>(3.14159265358979323846);

                mean = extraArguments[0];
                stddev = extraArguments[1];

                step = (blockDim.x * gridDim.x);
            }
            __syncthreads();

            // using this loop instead of memcpy
            for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e+= blockDim.x)
                cB[e] = dB[e];

            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            int middle = zLength % 2 == 0 ? zLength / 2 : zLength / 2 + 1;
            T t(-2.0f);

            for (int e = tid; e < middle; e += step) {
                auto epm = e + middle;
                // we need to get random values
                T r0 = rng->relativeT<T>(e, epsilon, static_cast<T>(1.0f));
                T r1 = rng->relativeT<T>(epm, epsilon, static_cast<T>(1.0f));

                T realMean0 = y == z ? mean : y[e * yEWS];

                z[e * zEWS] = (sd::math::nd4j_sqrt<T,T>(t * sd::math::nd4j_log<T,T>(r0)) * sd::math::nd4j_cos<T,T>(two_pi * r1)) * stddev + realMean0;

                if (epm < zLength) {
                    T realMean1 = y == z ? mean : y[epm * yEWS];
                    z[epm * zEWS] =  (sd::math::nd4j_sqrt<T,T>(t * sd::math::nd4j_log<T,T>(r0)) * sd::math::nd4j_sin<T,T>(two_pi * r1)) * stddev + realMean1;
                }
            }
        }
#endif


        static inline void
        specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            const T two_pi = static_cast<T>(2.0f) * static_cast<T>(3.14159265358979323846);

            auto zLength = shape::length(zShapeBuffer);
            auto yEWS = shape::elementWiseStride(yShapeBuffer);
            auto zEWS = shape::elementWiseStride(zShapeBuffer);

            auto middle = zLength % 2  + zLength / 2;

            int elementsPerThread = middle / TAD_THRESHOLD;
            int _threads = sd::math::nd4j_max<int>(1, elementsPerThread);
            _threads = sd::math::nd4j_min<int>(_threads, sd::Environment::getInstance()->maxThreads());

            int span = (middle / _threads) + 8;

            // we're enforcing even chunks, since it's mandatory for this algorithm
            span -= span % 2;

            //sd::random::RandomBuffer *buffer = reinterpret_cast<sd::random::RandomBuffer *> (state);
            sd::graph::RandomGenerator* rng = reinterpret_cast<sd::graph::RandomGenerator*>(state);
            const T mean = extraArguments[0];
            const T stddev = extraArguments[1];

            const T epsilon = static_cast<T>(1e-5);

            auto func = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e++) {
                    auto epm = e + middle;

                    // we need to get random values
                    T r0 = rng->relativeT<T>(e, epsilon, static_cast<T>(1.0f));
                    T r1 = rng->relativeT<T>(epm, epsilon, static_cast<T>(1.0f));

                    T realMean0 = y == z ? mean : y[e * yEWS];

                    auto z0 = (sd::math::nd4j_sqrt<T, T>(static_cast<T>(-2.0f) * sd::math::nd4j_log<T, T>(r0)) *
                               sd::math::nd4j_cos<T, T>(two_pi * r1)) * stddev + realMean0;
                    z[e * zEWS] = z0;

                    if (epm < zLength) {
                        T realMean1 = y == z ? mean : y[epm * yEWS];
                        auto z1 = (sd::math::nd4j_sqrt<T, T>(static_cast<T>(-2.0f) * sd::math::nd4j_log<T, T>(r0)) *
                                   sd::math::nd4j_sin<T, T>(two_pi * r1)) * stddev + realMean1;
                        z[epm * zEWS] = z1;
                    }
                }
            };

            sd::Threads::parallel_for(func, 0, middle, 1, _threads);
        }
    };


//////////////////////////////////////////////////////////////////////
    /**
    * This Op produces random values within [0..N], Distribuion is binomial
    */
    template<typename T>
    class BinomialDistribution {
    public:


        method_XY
        method_X
        method_idx

        static const bool requiresSpecial = true;

#ifdef __CUDACC__
        __device__ static inline void specialOpCuda(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            int trials = (int) extraArguments[0];
            T prob = extraArguments[1];

            __shared__ Nd4jLong zLength;
            __shared__ int yEWS;
            __shared__ int zEWS;

            __shared__ sd::graph::RandomGenerator* rng;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ sd::graph::RandomGenerator *devRng;
            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                rng = reinterpret_cast<sd::graph::RandomGenerator*>(shmem);
                cB = shmem;
                devRng = reinterpret_cast<sd::graph::RandomGenerator*>(state);
                dB = reinterpret_cast<unsigned char *> (state);

                zLength = shape::length(zShapeBuffer);
                yEWS = shape::elementWiseStride(yShapeBuffer);
                zEWS = shape::elementWiseStride(zShapeBuffer);
            }
            __syncthreads();

            // using this loop instead of memcpy
            for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e+= blockDim.x)
                cB[e] = dB[e];

            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            for (Nd4jLong e = tid; e < zLength; e += blockDim.x * gridDim.x) {
                int success = 0;
                for (int t = 1; t <= trials; t++) {
                    T randVal = rng->relativeT<T>((e+1) * t);
                    if (y != z) {
                        // we're using external probs
                        prob = y[(t-1) * yEWS];
                    }

                    if (randVal < prob)
                        success++;
                }
                // if trials is set to 0, effectively we just have successful memset
                z[e * zEWS] = static_cast<T>(success);
            }
        }
#endif

        static inline void specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            int trials = (int) extraArguments[0];

            Nd4jLong zLength = shape::length(zShapeBuffer);

            auto yEWS = shape::elementWiseStride(yShapeBuffer);
            auto zEWS = shape::elementWiseStride(zShapeBuffer);

            int elementsPerThread = zLength / TAD_THRESHOLD;
            int _threads = sd::math::nd4j_max<int>(1, elementsPerThread);
            _threads = sd::math::nd4j_min<int>(_threads, sd::Environment::getInstance()->maxThreads());

            T prob = extraArguments[1];

            sd::graph::RandomGenerator* rng = reinterpret_cast<sd::graph::RandomGenerator*>(state);
            auto func = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e++) {

                    int success = 0;
                    for (int t = 1; t <= trials; t++) {
                        T randVal = rng->relativeT<T>((e+1) * t);
                        if (y != z) {
                            // we're using external probs
                            prob = y[(t-1) * yEWS];
                        }

                        if (randVal < prob)
                            success++;
                    }

                    // if trials is set to 0, effectively we just have successful memset
                    z[e * zEWS] = static_cast<T>(success);
                }
            };

            sd::Threads::parallel_for(func, 0, zLength, 1, _threads);
        }
    };


//////////////////////////////////////////////////////////////////////
    /**
    * This Op produces random values within [0..N], Distribuion is binomial
    */
    template<typename T>
    class BinomialDistributionEx {
    public:


        method_XY
        method_X
        method_idx

        static const bool requiresSpecial = true;

#ifdef __CUDACC__
        __device__ static inline void specialOpCuda(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            int trials = (int) extraArguments[0];
            T prob = extraArguments[1];

            __shared__ Nd4jLong zLength;
            __shared__ int yEWS;
            __shared__ int zEWS;

            __shared__ sd::graph::RandomGenerator* rng;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ sd::graph::RandomGenerator *devRng;
            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                rng = (sd::graph::RandomGenerator*) shmem;
                cB = shmem;
                devRng = reinterpret_cast<sd::graph::RandomGenerator*> (state);
                dB = reinterpret_cast<unsigned char *> (state);

                zLength = shape::length(zShapeBuffer);
                yEWS = shape::elementWiseStride(yShapeBuffer);
                zEWS = shape::elementWiseStride(zShapeBuffer);
            }
            __syncthreads();

            // using this loop instead of memcpy
            for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e+= blockDim.x)
                cB[e] = dB[e];

            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            for (Nd4jLong e = tid; e < zLength; e += blockDim.x * gridDim.x) {
                int success = 0;
                for (int t = 1; t <= trials; t++) {
                    T randVal = rng->relativeT<T>((e+1) * t);
                    if (y != z) {
                        // we're using external probs
                        prob = y[e * yEWS];
                    }

                    if (randVal < prob)
                        success++;
                }

                // if trials is set to 0, effectively we just have successful memset
                z[e * zEWS] = (T) success;
            }
        }
#endif

        static inline void specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            int trials = (int) extraArguments[0];

            Nd4jLong zLength = shape::length(zShapeBuffer);

            auto yEWS = shape::elementWiseStride(yShapeBuffer);
            auto zEWS = shape::elementWiseStride(zShapeBuffer);

            int elementsPerThread = zLength / TAD_THRESHOLD;
            int _threads = sd::math::nd4j_max<int>(1, elementsPerThread);
            _threads = sd::math::nd4j_min<int>(_threads, sd::Environment::getInstance()->maxThreads());

            T prob = extraArguments[1];

            //sd::random::RandomBuffer *buffer = reinterpret_cast<sd::random::RandomBuffer *> (state);
            sd::graph::RandomGenerator* rng = reinterpret_cast<sd::graph::RandomGenerator*>(state);
            auto func = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e++) {

                    int success = 0;
                    for (int t = 1; t <= trials; t++) {
                        T randVal = rng->relativeT<T>((e+1) * t);
                        if (y != z) {
                            // we're using external probs
                            prob = y[e * yEWS];
                        }

                        if (randVal < prob)
                            success++;
                    }

                    // if trials is set to 0, effectively we just have successful memset
                    z[e * zEWS] = static_cast<T>(success);
                }
            };

            sd::Threads::parallel_for(func, 0, zLength, 1, _threads);
        }
    };

//////////////////////////////////////////////////////////////////////
    // This Op produces random Gaussian values within [mean-2*stddev,mean+2*stddev]
    template<typename T>
    class TruncatedNormalDistribution {
    private:
        static inline _CUDA_HD T step(sd::graph::RandomGenerator* rng, T mean, T stddev, Nd4jLong e, Nd4jLong middle, T& z) {
            auto epm = e + middle;
            const T two_pi = static_cast<T>(2.0f) * static_cast<T>(3.14159265358979323846);
            const T epsilon = static_cast<T>(1.e-5f);
            // we need to get random values
            T r0 = rng->relativeT<T>(e, epsilon, static_cast<T>(1.0f));
            T r1 = rng->relativeT<T>(epm, epsilon, static_cast<T>(1.0f));

            T realMean0 = mean;

            auto z0 =  (sd::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * sd::math::nd4j_log<T,T>(r0)) * sd::math::nd4j_cos<T,T>(two_pi * r1)) * stddev + realMean0;
            z = z0;
            if (epm < middle) {
                T realMean1 = mean;
                auto z1 = (sd::math::nd4j_sqrt<T, T>(static_cast<T>(-2.0f) * sd::math::nd4j_log<T, T>(r0)) *
                           sd::math::nd4j_sin<T, T>(two_pi * r1)) * stddev + realMean1;
                z = z1;
            }
            return z;
        }
    public:

        method_XY
        method_X
        method_idx

        static const bool requiresSpecial = true;

#ifdef __CUDACC__
        __device__ static inline void specialOpCuda(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            __shared__ T epsilon;
            __shared__ T two_pi;

            __shared__ Nd4jLong zLength;
            __shared__ Nd4jLong zEWS;
            __shared__ Nd4jLong yEWS;
            __shared__ T mean;
            __shared__ T stddev;
            __shared__ int step;

            __shared__ T *tZ;

            __shared__ sd::graph::RandomGenerator* rng;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ sd::graph::RandomGenerator* devRng;
            __shared__ Nd4jLong middle;

            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                rng = reinterpret_cast<sd::graph::RandomGenerator*>(shmem);
                cB = shmem;
                devRng = reinterpret_cast<sd::graph::RandomGenerator*> (state);
                dB = reinterpret_cast<unsigned char *> (state);

                tZ = reinterpret_cast<T*>(shmem + sizeof(sd::graph::RandomGenerator));

                zLength = shape::length(zShapeBuffer);
                zEWS = shape::elementWiseStride(zShapeBuffer);
                yEWS = shape::elementWiseStride(yShapeBuffer);

                epsilon = static_cast<T>(1e-6f);
                two_pi = static_cast<T>(2.0f) * static_cast<T>(3.14159265358979323846);

                mean = extraArguments[0];
                stddev = extraArguments[1];

                step = (blockDim.x * gridDim.x);
                middle = zLength / 2 + (zLength % 2);
            }
            __syncthreads();

            // using this loop instead of memcpy
            for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e+= blockDim.x)
                cB[e] = dB[e];

            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            GaussianDistribution<T>::specialOpCuda(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
            __syncthreads();

            T ds = sd::math::nd4j_abs<T>(stddev) * static_cast<T>(2.0f);
            for (Nd4jLong e = tid; e < zLength; e += step) {
                if (z[e] > mean + ds || z[e] < mean - ds) {
                    z[e] = TruncatedNormalDistribution<T>::step(rng, mean, stddev, e, middle, z[e]);

                    if (z[e] > mean + ds || z[e] < mean - ds)
                        z[e] = mean + sd::DataTypeUtils::min<T>();
                }
            }
        }
#endif

        static inline void
        specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            GaussianDistribution<T>::specialOp(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
            Nd4jLong zLength = shape::length(zShapeBuffer);
            //auto yEWS = shape::elementWiseStride(yShapeBuffer);
            //auto zEWS = shape::elementWiseStride(zShapeBuffer);
            sd::graph::RandomGenerator* rng = reinterpret_cast<sd::graph::RandomGenerator*>(state);
            T mean = extraArguments[0];
            T stddev = extraArguments[1];
            T ds = sd::math::nd4j_abs<T>(stddev) * (T) 2.0f;
            Nd4jLong middle = zLength / 2 + (zLength % 2);
            int elementsPerThread = middle / TAD_THRESHOLD;
            int _threads = sd::math::nd4j_max<int>(1, elementsPerThread);
            _threads = sd::math::nd4j_min<int>(_threads, sd::Environment::getInstance()->maxThreads());

            const T epsilon = static_cast<T>(1e-5);

            auto func = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e++) {
                    if (z[e] > mean + ds || z[e] < mean - ds) {
                        z[e] = step(rng, mean, stddev, e, middle, z[e]);

                        if (z[e] > mean + ds || z[e] < mean - ds)
                            z[e] = mean + sd::DataTypeUtils::min<T>();
                    }
                }
            };

            sd::Threads::parallel_for(func, 0, zLength, 1, _threads);
        }
    };

//////////////////////////////////////////////////////////////////////
// This Op produces random Log-normal distribution
 template<typename T>
    class LogNormalDistribution {
    public:

        method_XY
        method_X
        method_idx

        static const bool requiresSpecial = true;


#ifdef __CUDACC__
        __device__ static inline void specialOpCuda(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            __shared__ T epsilon;
            __shared__ T two_pi;

            __shared__ Nd4jLong zLength;
            __shared__ Nd4jLong zEWS;
            __shared__ Nd4jLong yEWS;
            __shared__ T mean;
            __shared__ T stddev;
            __shared__ int step;

            __shared__ T *tZ;

            __shared__ sd::graph::RandomGenerator* rng;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ sd::graph::RandomGenerator* devRng;

            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                rng = reinterpret_cast<sd::graph::RandomGenerator*>(state);
                cB = shmem;
                devRng = reinterpret_cast<sd::graph::RandomGenerator*>(state);

                dB = reinterpret_cast<unsigned char *> (state);

                tZ = reinterpret_cast<T*>(shmem + sizeof(sd::graph::RandomGenerator));

                zLength = shape::length(zShapeBuffer);
                zEWS = shape::elementWiseStride(zShapeBuffer);
                yEWS = shape::elementWiseStride(yShapeBuffer);


                epsilon = static_cast<T>(1e-5);
                two_pi = static_cast<T>(2.0f) * static_cast<T>(3.14159265358979323846);

                mean = extraArguments[0];
                stddev = extraArguments[1];

                step = (blockDim.x * gridDim.x);
            }
            __syncthreads();

            // using this loop instead of memcpy
            for (int e = threadIdx.x; e < sizeof(sd::graph::RandomGenerator); e+= blockDim.x)
                cB[e] = dB[e];

            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            int middle = zLength % 2 == 0 ? zLength / 2 : zLength / 2 + 1;

            for (Nd4jLong e = tid; e < middle; e += step) {
                auto epm = e + middle;

                // we need to get random values
                T r0 = rng->relativeT<T>(e, epsilon, static_cast<T>(1.0f));
                T r1 = rng->relativeT<T>(epm, epsilon, static_cast<T>(1.0f));

                T realMean = y == z ? mean : y[e * yEWS];

                z[e *zEWS] =  sd::math::nd4j_exp<T,T>((sd::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * sd::math::nd4j_log<T,T>(r0)) * sd::math::nd4j_cos<T,T>(two_pi * r1)) * stddev + realMean);

                if (epm < zLength) {
                    realMean = y == z ? mean : y[epm * yEWS];
                    z[epm *zEWS] =  sd::math::nd4j_exp<T,T>((sd::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * sd::math::nd4j_log<T,T>(r0)) * sd::math::nd4j_sin<T,T>(two_pi * r1)) * stddev + realMean);
                }
            }
        }
#endif

        static inline void
        specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            const T two_pi = static_cast<T>(2.0f) * static_cast<T>(3.14159265358979323846);

            Nd4jLong zLength = shape::length(zShapeBuffer);
            auto yEWS = shape::elementWiseStride(yShapeBuffer);
            auto zEWS = shape::elementWiseStride(zShapeBuffer);

            auto middle = zLength % 2 == 0 ? zLength / 2 : zLength / 2 + 1;

            int elementsPerThread = middle / TAD_THRESHOLD;
            int _threads = sd::math::nd4j_max<int>(1, elementsPerThread);
            _threads = sd::math::nd4j_min<int>(_threads, sd::Environment::getInstance()->maxThreads());

            int span = (zLength / _threads) + 8;

            // we're enforcing even chunks, since it's mandatory for this algorithm
            span -= span % 2;

//            auto buffer = reinterpret_cast<sd::random::RandomBuffer *> (state);
            sd::graph::RandomGenerator* rng = reinterpret_cast<sd::graph::RandomGenerator*>(state);

            const T mean = extraArguments[0];
            const T stddev = extraArguments[1];
            const T epsilon = static_cast<T>(1e-5);

            auto func = PRAGMA_THREADS_FOR {
                PRAGMA_OMP_SIMD
                for (auto e = start; e < stop; e++) {
                    auto epm = e + middle;

                    // we need to get random values
                    T r0 = rng->relativeT<T>(e, epsilon, static_cast<T>(1.0f));
                    T r1 = rng->relativeT<T>(epm, epsilon, static_cast<T>(1.0f));

                    T realMean = y == z ? mean : y[e * yEWS];

                    z[e * zEWS] =  sd::math::nd4j_exp<T,T>((sd::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * sd::math::nd4j_log<T,T>(r0)) * sd::math::nd4j_cos<T,T>(two_pi * r1)) * stddev + realMean);

                    if (epm < zLength) {
                        realMean = y == z ? mean : y[epm * yEWS];
                        z[epm * zEWS] =  sd::math::nd4j_exp<T,T>((sd::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * sd::math::nd4j_log<T,T>(r0)) * sd::math::nd4j_sin<T,T>(two_pi * r1)) * stddev + realMean);
                    }
                }
            };

            sd::Threads::parallel_for(func, 0, middle, 1, _threads);
        }
    };


}

#endif //LIBND4J_SPECIAL_RANDOM_OPS_H
