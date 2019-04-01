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

            __shared__ nd4j::graph::RandomGenerator *rng;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ nd4j::graph::RandomGenerator *devRng;
            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                rng = (nd4j::graph::RandomGenerator*) shmem;
                cB = shmem;
                devRng = reinterpret_cast<nd4j::graph::RandomGenerator*> (state);
                dB = reinterpret_cast<unsigned char *> (state);

                xLength = shape::length(xShapeBuffer);
                yLength = shape::length(yShapeBuffer);
                zLength = shape::length(zShapeBuffer);

                xEWS = shape::elementWiseStride(xShapeBuffer);
                yEWS = shape::elementWiseStride(yShapeBuffer);
                zEWS = shape::elementWiseStride(zShapeBuffer);
            }
            __syncthreads();

            // using this loop instead of memcpy
            for (int e = threadIdx.x; e < sizeof(nd4j::graph::RandomGenerator); e+= blockDim.x) {
                cB[e] = dB[e];
            }
            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (zEWS >= 1 && xEWS >= 1 && yEWS >= 1) {
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
                    }
                }
            } 
            else {
            
                for (Nd4jLong i = tid; i < zLength; i+=blockDim.x * gridDim.x) {

                    auto zOffset2 = shape::getIndexOffset(i, zShapeBuffer, zLength);
                    T prob = rng->relativeT<T>(i);
                    T cumProb = (T) 0.0f;

                    for (Nd4jLong f = 0; f < yLength; f++) {
                        
                        auto yOffset2 = shape::getIndexOffset(f, yShapeBuffer, yLength);
                        T relProb = y[yOffset2];
                        cumProb += relProb;

                        if (prob <= cumProb || f == yLength - 1) {                            
                            
                            auto xOffset2 = shape::getIndexOffset(f, xShapeBuffer, xLength);
                            z[zOffset2] = x[xOffset2];
                            f += yLength;
                        }
                    }
                }
            }

            __syncthreads();
            if (threadIdx.x == 0 && blockIdx.x == 0)
                devRng->rewindH(zLength);
        }
#endif

        static inline void specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            /**
             * X holds data,
             * Y holds probabilities
             * Z will hold results
             */

            //nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
            nd4j::graph::RandomGenerator* rng = reinterpret_cast<nd4j::graph::RandomGenerator*>(state);
            // TODO: we probably might want to skip this sum, and state that probabilities array should be real probabilities, i.e. should sum to 1.0
            //T probSum = extraArguments[0];

            Nd4jLong xLength = shape::length(xShapeBuffer);
            Nd4jLong yLength = shape::length(yShapeBuffer);
            Nd4jLong zLength = shape::length(zShapeBuffer);

            auto xEWS = shape::elementWiseStride(xShapeBuffer);
            auto yEWS = shape::elementWiseStride(yShapeBuffer);
            auto zEWS = shape::elementWiseStride(zShapeBuffer);

            int elementsPerThread = zLength / TAD_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            if (zEWS >= 1 && xEWS >= 1 && yEWS >= 1) {
                PRAGMA_OMP_PARALLEL_FOR_THREADS(_threads)
                for (Nd4jLong e = 0; e < zLength; e++) {
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
            } 
            else {

                PRAGMA_OMP_PARALLEL_FOR_THREADS(_threads)
                for (Nd4jLong i = 0; i < zLength; i++) {

                    auto zOffset2 = shape::getIndexOffset(i, zShapeBuffer, zLength);
                    T prob = rng->relativeT<T>(i);
                    T cumProb = (T) 0.0f;

                    for (Nd4jLong f = 0; f < yLength; f++) {
                        
                        auto yOffset2 = shape::getIndexOffset(f, yShapeBuffer, yLength);
                        T relProb = y[yOffset2];
                        cumProb += relProb;

                        if (prob <= cumProb || f == yLength - 1) {                        
                            
                            auto xOffset2 = shape::getIndexOffset(f, xShapeBuffer, xLength);
                            z[zOffset2] = x[xOffset2];
                            break;
                        }
                    }
                }
            }

            // update rng state
            rng->rewindH(zLength);
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

            __shared__ nd4j::graph::RandomGenerator* rng;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ nd4j::graph::RandomGenerator *devRng;

            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                rng = reinterpret_cast<nd4j::graph::RandomGenerator*>(shmem);
                cB = shmem;
                devRng = reinterpret_cast<nd4j::graph::RandomGenerator *> (state);
                dB = reinterpret_cast<unsigned char *> (state);

                tZ = reinterpret_cast<T *>(shmem + sizeof(nd4j::graph::RandomGenerator));

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
            for (int e = threadIdx.x; e < sizeof(nd4j::graph::RandomGenerator); e+= blockDim.x) {
                cB[e] = dB[e];
            }
            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            int middle = zLength % 2 == 0 ? zLength / 2 : zLength / 2 + 1;

            for (int e = tid; e < middle; e += step) {
                auto epm = e + middle;

                // we need to get random values
                T r0 = rng->relativeT<T>(e, epsilon, static_cast<T>(1.0f));
                T r1 = rng->relativeT<T>(epm, epsilon, static_cast<T>(1.0f));

                T realMean0 = y == z ? mean : y[e * yEWS];

                z[e * zEWS] =  (nd4j::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T,T>(r0)) * nd4j::math::nd4j_cos<T,T>(two_pi * r1)) * stddev + realMean0;

                if (epm < zLength) {
                    T realMean1 = y == z ? mean : y[epm * yEWS];
                    z[epm * zEWS] =  (nd4j::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T,T>(r0)) * nd4j::math::nd4j_sin<T,T>(two_pi * r1)) * stddev + realMean1;
                }
            }

            __syncthreads();

            if (threadIdx.x == 0 && blockIdx.x == 0)
                devRng->rewindH(zLength);
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
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            int span = (middle / _threads) + 8;

            // we're enforcing even chunks, since it's mandatory for this algorithm
            span -= span % 2;

            //nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
            nd4j::graph::RandomGenerator* rng = reinterpret_cast<nd4j::graph::RandomGenerator*>(state);
            const T mean = extraArguments[0];
            const T stddev = extraArguments[1];

            const T epsilon = static_cast<T>(1e-5);

            PRAGMA_OMP_PARALLEL_FOR_THREADS(_threads)
            for (Nd4jLong e = 0; e < middle; e++) {
                auto epm = e + middle;

                // we need to get random values
                T r0 = rng->relativeT<T>(e, epsilon, static_cast<T>(1.0f));
                T r1 = rng->relativeT<T>(epm, epsilon, static_cast<T>(1.0f));

                T realMean0 = y == z ? mean : y[e * yEWS];

                auto z0 =  (nd4j::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T,T>(r0)) * nd4j::math::nd4j_cos<T,T>(two_pi * r1)) * stddev + realMean0;
                z[e * zEWS] = z0;

                if (epm < zLength) {
                    T realMean1 = y == z ? mean : y[epm * yEWS];
                    auto z1 = (nd4j::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T,T>(r0)) * nd4j::math::nd4j_sin<T,T>(two_pi * r1)) * stddev + realMean1;
                    z[epm * zEWS] = z1;
                }
            }

            // update rng state
            rng->rewindH(zLength);
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

            __shared__ nd4j::graph::RandomGenerator* rng;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ nd4j::graph::RandomGenerator *devRng;
            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                rng = reinterpret_cast<nd4j::graph::RandomGenerator*>(shmem);
                cB = shmem;
                devRng = reinterpret_cast<nd4j::graph::RandomGenerator*>(state);
                dB = reinterpret_cast<unsigned char *> (state);

                zLength = shape::length(zShapeBuffer);
                yEWS = shape::elementWiseStride(yShapeBuffer);
                zEWS = shape::elementWiseStride(zShapeBuffer);
            }
            __syncthreads();

            // using this loop instead of memcpy
            for (int e = threadIdx.x; e < sizeof(nd4j::graph::RandomGenerator); e+= blockDim.x) {
                cB[e] = dB[e];
            }
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

            __syncthreads();
            if (trials > 0 && threadIdx.x == 0 && blockIdx.x == 0)
                devRng->rewindH(zLength * trials);
        }
#endif

        static inline void specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            int trials = (int) extraArguments[0];

            Nd4jLong zLength = shape::length(zShapeBuffer);

            auto yEWS = shape::elementWiseStride(yShapeBuffer);
            auto zEWS = shape::elementWiseStride(zShapeBuffer);

            int elementsPerThread = zLength / TAD_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            auto span = (zLength / _threads) + 8;

            nd4j::graph::RandomGenerator* rng = reinterpret_cast<nd4j::graph::RandomGenerator*>(state);
            PRAGMA_OMP_PARALLEL_THREADS(_threads)
            {
                int tid = omp_get_thread_num();
                auto start = span * tid;
                auto end = span * (tid + 1);
                if (end > zLength) end = zLength;

                T prob = extraArguments[1];

                for (Nd4jLong e = start; e < end; e++) {

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

            // update rng state
            if (trials > 0)
                rng->rewindH(zLength * trials);
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

            __shared__ nd4j::graph::RandomGenerator* rng;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ nd4j::graph::RandomGenerator *devRng;
            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                rng = (nd4j::graph::RandomGenerator*) shmem;
                cB = shmem;
                devRng = reinterpret_cast<nd4j::graph::RandomGenerator*> (state);
                dB = reinterpret_cast<unsigned char *> (state);

                zLength = shape::length(zShapeBuffer);
                yEWS = shape::elementWiseStride(yShapeBuffer);
                zEWS = shape::elementWiseStride(zShapeBuffer);
            }
            __syncthreads();

            // using this loop instead of memcpy
            for (int e = threadIdx.x; e < sizeof(nd4j::graph::RandomGenerator); e+= blockDim.x) {
                cB[e] = dB[e];
            }
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

            __syncthreads();
             if (trials > 0 && threadIdx.x == 0 && blockIdx.x == 0)
                 devRng->rewindH(zLength * trials);
        }
#endif

        static inline void specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            int trials = (int) extraArguments[0];

            Nd4jLong zLength = shape::length(zShapeBuffer);

            auto yEWS = shape::elementWiseStride(yShapeBuffer);
            auto zEWS = shape::elementWiseStride(zShapeBuffer);

            int elementsPerThread = zLength / TAD_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            auto span = (zLength / _threads) + 8;

            //nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
            nd4j::graph::RandomGenerator* rng = reinterpret_cast<nd4j::graph::RandomGenerator*>(state);
            PRAGMA_OMP_PARALLEL_THREADS(_threads)
            {
                int tid = omp_get_thread_num();
                Nd4jLong start = span * tid;
                Nd4jLong end = span * (tid + 1);
                if (end > zLength) end = zLength;

                T prob = extraArguments[1];

                for (Nd4jLong e = start; e < end; e++) {

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
            }

            // update rng state
            if (trials > 0)
                rng->rewindH(zLength * trials);
        }
    };
    
//////////////////////////////////////////////////////////////////////        
    // This Op produces random Gaussian values within [mean-2*stddev,mean+2*stddev]
    template<typename T>
    class TruncatedNormalDistribution {
    private:
        static T step(nd4j::graph::RandomGenerator* rng, T mean, T stddev, Nd4jLong e, Nd4jLong middle, T& z) {
            auto epm = e + middle;
            const T two_pi = static_cast<T>(2.0f) * static_cast<T>(3.14159265358979323846);
            const T epsilon = static_cast<T>(1.e-5f);
            // we need to get random values
            T r0 = rng->relativeT<T>(e, epsilon, static_cast<T>(1.0f));
            T r1 = rng->relativeT<T>(epm, epsilon, static_cast<T>(1.0f));

            T realMean0 = mean;

            auto z0 =  (nd4j::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T,T>(r0)) * nd4j::math::nd4j_cos<T,T>(two_pi * r1)) * stddev + realMean0;
            z = z0;
            if (epm < middle) {
                T realMean1 = mean;
                auto z1 = (nd4j::math::nd4j_sqrt<T, T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T, T>(r0)) *
                           nd4j::math::nd4j_sin<T, T>(two_pi * r1)) * stddev + realMean1;
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

            __shared__ nd4j::graph::RandomGenerator* rng;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ nd4j::graph::RandomGenerator* devRng;

            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                rng = reinterpret_cast<nd4j::graph::RandomGenerator*>(shmem);
                cB = shmem;
                devRng = reinterpret_cast<nd4j::graph::RandomGenerator*> (state);
                dB = reinterpret_cast<unsigned char *> (state);

                tZ = reinterpret_cast<T*>(shmem + sizeof(nd4j::graph::RandomGenerator));

                zLength = shape::length(zShapeBuffer);
                zEWS = shape::elementWiseStride(zShapeBuffer);
                yEWS = shape::elementWiseStride(yShapeBuffer);


                epsilon = static_cast<T>(1e-6f);
                two_pi = static_cast<T>(2.0f) * static_cast<T>(3.14159265358979323846);

                mean = extraArguments[0];
                stddev = extraArguments[1];

                step = (blockDim.x * gridDim.x);
            }
            __syncthreads();

            // using this loop instead of memcpy
            for (int e = threadIdx.x; e < sizeof(nd4j::graph::RandomGenerator); e+= blockDim.x) {
                cB[e] = dB[e];
            }
            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int middle = zLength % 2 == 0 ? zLength / 2 : zLength / 2 + 1;
            T result0, result1, u0, u1, z0, z1, uT, uP;

            T ds = nd4j::math::nd4j_abs<T>(stddev) * static_cast<T>(2.0f);
            for (Nd4jLong e = tid; e < middle; e += step) {
                // we need to get random values

                Nd4jLong generation0 = 0;
                auto epm = e + middle;
                T realMean0 = y == z ? mean : y[e * yEWS];
                T realMean1 = y == z ? mean : y[epm * yEWS];
                T aRealMean0 = nd4j::math::nd4j_abs<T>(realMean0);
                T aRealMean1 = nd4j::math::nd4j_abs<T>(realMean1);

                do {
                    u0 = rng->relativeT<T>(e + generation0, epsilon, static_cast<T>(1.0f));
                    u1 = rng->relativeT<T>(epm + generation0, epsilon, static_cast<T>(1.0f));

                    uT = nd4j::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T,T>(u0));
                    uP = two_pi * u1;

                    z0 = uT * nd4j::math::nd4j_cos<T,T>(uP);
                    z1 = uT * nd4j::math::nd4j_sin<T,T>(uP);

                    result0 = z0 * stddev + realMean0;
                    result1 = z1 * stddev + realMean1;

                    generation0 += zLength;
                } while (ds < aRealMean0 + nd4j::math::nd4j_abs<T>(result0) || aRealMean1 + nd4j::math::nd4j_abs<T>(result1) > ds);

                z[e * zEWS] = result0;
                if((epm) < zLength)
                    z[epm * zEWS] = result1;
            }

            __syncthreads();
            if (threadIdx.x == 0 && blockIdx.x == 0)
                devRng->rewindH(zLength);
        }
#endif

        static inline void
        specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            GaussianDistribution<T>::specialOp(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
            Nd4jLong zLength = shape::length(zShapeBuffer);
            //auto yEWS = shape::elementWiseStride(yShapeBuffer);
            //auto zEWS = shape::elementWiseStride(zShapeBuffer);
            nd4j::graph::RandomGenerator* rng = reinterpret_cast<nd4j::graph::RandomGenerator*>(state);
            T mean = extraArguments[0];
            T stddev = extraArguments[1];
            T ds = nd4j::math::nd4j_abs<T>(stddev) * (T) 2.0f;
            Nd4jLong middle = zLength / 2 + (zLength % 2);
            int elementsPerThread = middle / TAD_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            const T epsilon = static_cast<T>(1e-5);

            PRAGMA_OMP_PARALLEL_FOR_THREADS(_threads)
            for (Nd4jLong e = 0; e < zLength; ++e) {
                if (z[e] > mean + ds || z[e] < mean - ds) {
                    z[e] = step(rng, mean, stddev, e, middle, z[e]);// = e > 0 ? z[e - 1] : mean; // + stddev;

                //else if (z[e] < mean - ds)
                    if (z[e] > mean + ds || z[e] < mean - ds)
                        z[e] = mean + nd4j::DataTypeUtils::min<T>();
                }
            }

            // update rng state
            rng->rewindH(zLength);
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

            __shared__ nd4j::graph::RandomGenerator* rng;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ nd4j::graph::RandomGenerator* devRng;

            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                rng = reinterpret_cast<nd4j::graph::RandomGenerator*>(state);
                cB = shmem;
                devRng = reinterpret_cast<nd4j::graph::RandomGenerator*>(state);

                dB = reinterpret_cast<unsigned char *> (state);

                tZ = reinterpret_cast<T*>(shmem + sizeof(nd4j::graph::RandomGenerator));

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
            for (int e = threadIdx.x; e < sizeof(nd4j::graph::RandomGenerator); e+= blockDim.x) {
                cB[e] = dB[e];
            }
            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            int middle = zLength % 2 == 0 ? zLength / 2 : zLength / 2 + 1;

            for (Nd4jLong e = tid; e < middle; e += step) {
                auto epm = e + middle;

                // we need to get random values
                T r0 = rng->relativeT<T>(e, epsilon, static_cast<T>(1.0f));
                T r1 = rng->relativeT<T>(epm, epsilon, static_cast<T>(1.0f));

                T realMean = y == z ? mean : y[e * yEWS];

                z[e *zEWS] =  nd4j::math::nd4j_exp<T,T>((nd4j::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T,T>(r0)) * nd4j::math::nd4j_cos<T,T>(two_pi * r1)) * stddev + realMean);

                if (epm < zLength) {
                    realMean = y == z ? mean : y[epm * yEWS];
                    z[epm *zEWS] =  nd4j::math::nd4j_exp<T,T>((nd4j::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T,T>(r0)) * nd4j::math::nd4j_sin<T,T>(two_pi * r1)) * stddev + realMean);
                }
            }

            __syncthreads();
            if (threadIdx.x == 0 && blockIdx.x == 0)
                devRng->rewindH(zLength);
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
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            int span = (zLength / _threads) + 8;

            // we're enforcing even chunks, since it's mandatory for this algorithm
            span -= span % 2;

//            auto buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
            nd4j::graph::RandomGenerator* rng = reinterpret_cast<nd4j::graph::RandomGenerator*>(state);

            const T mean = extraArguments[0];
            const T stddev = extraArguments[1];
            const T epsilon = static_cast<T>(1e-5);

            PRAGMA_OMP_PARALLEL_THREADS(_threads)
            {
                int tid = omp_get_thread_num();
                Nd4jLong start = span * tid;
                Nd4jLong end = span * (tid + 1);
                if (end > middle)
                    end = middle;

                PRAGMA_OMP_SIMD
                for (Nd4jLong e = start; e < end; e++) {
                    auto epm = e + middle;

                    // we need to get random values
                    T r0 = rng->relativeT<T>(e, epsilon, static_cast<T>(1.0f));
                    T r1 = rng->relativeT<T>(epm, epsilon, static_cast<T>(1.0f));

                    T realMean = y == z ? mean : y[e * yEWS];

                    z[e * zEWS] =  nd4j::math::nd4j_exp<T,T>((nd4j::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T,T>(r0)) * nd4j::math::nd4j_cos<T,T>(two_pi * r1)) * stddev + realMean);

                    if (epm < zLength) {
                        realMean = y == z ? mean : y[epm * yEWS];
                        z[epm * zEWS] =  nd4j::math::nd4j_exp<T,T>((nd4j::math::nd4j_sqrt<T,T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T,T>(r0)) * nd4j::math::nd4j_sin<T,T>(two_pi * r1)) * stddev + realMean);
                    }
                }
            }

            // update rng state
            rng->rewindH(zLength);

        }
    };


}

#endif //LIBND4J_SPECIAL_RANDOM_OPS_H
