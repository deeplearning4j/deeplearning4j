//
// @author raver119@gmail.com
//

#ifndef LIBND4J_SPECIAL_RANDOM_OPS_H
#define LIBND4J_SPECIAL_RANDOM_OPS_H

#include <ops/random_ops.h>
#include <helpers/shape.h>

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

            __shared__ nd4j::random::RandomBuffer *buffer;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ nd4j::random::RandomBuffer *devBuffer;
            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                buffer = (nd4j::random::RandomBuffer *) shmem;
                cB = shmem;
                devBuffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
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
            for (int e = threadIdx.x; e < sizeof(nd4j::random::RandomBuffer); e+= blockDim.x) {
                cB[e] = dB[e];
            }
            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (zEWS >= 1 && xEWS >= 1 && yEWS >= 1) {
                for (Nd4jLong e = tid; e < zLength; e+=blockDim.x * gridDim.x) {
                    T prob = buffer->relativeT<T>(e);
                    T cumProb = (T) 0.0f;
                    for (Nd4jLong f = 0; f < yLength; f++) {
                        T relProb = y[f * yEWS];
                        cumProb += relProb;

                        if (prob <= cumProb || f == yLength - 1) {
                            z[e * zEWS] = x[f * xEWS];
                            f += yLength;
                        }
                        __syncthreads();
                    }
                    __syncthreads();
                }
            } else {
                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];
                Nd4jLong zCoord[MAX_RANK];

                __shared__ int xRank;
                __shared__ int yRank;
                __shared__ int zRank;

                __shared__ Nd4jLong *xShape;
                __shared__ Nd4jLong *yShape;
                __shared__ Nd4jLong *zShape;

                __shared__ Nd4jLong *xStride;
                __shared__ Nd4jLong *yStride;
                __shared__ Nd4jLong *zStride;


                if (threadIdx.x == 0) {
                    xRank = shape::rank(xShapeBuffer);
                    yRank = shape::rank(yShapeBuffer);
                    zRank = shape::rank(zShapeBuffer);

                    xShape = shape::shapeOf(xShapeBuffer);
                    yShape = shape::shapeOf(yShapeBuffer);
                    zShape = shape::shapeOf(zShapeBuffer);

                    xStride = shape::stride(xShapeBuffer);
                    yStride = shape::stride(yShapeBuffer);
                    zStride = shape::stride(zShapeBuffer);
                }
                __syncthreads();

                for (Nd4jLong i = tid; i < zLength; i+=blockDim.x * gridDim.x) {
                    shape::ind2sub(zRank, zShape, i, zCoord);

                    auto zOffset2 = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                    T prob = buffer->relativeT<T>(i);
                    T cumProb = (T) 0.0f;
                    for (Nd4jLong f = 0; f < yLength; f++) {
                        shape::ind2sub(yRank, yShape, i, yCoord);
                        auto yOffset2 = shape::getOffset(0, yShape, yStride, yCoord, yRank);

                        T relProb = y[yOffset2];
                        cumProb += relProb;

                        if (prob <= cumProb || f == yLength - 1) {
                            shape::ind2sub(xRank, xShape, f, xCoord);
                            auto xOffset2 = shape::getOffset(0, xShape, xStride, xCoord, xRank);

                            z[zOffset2] = x[xOffset2];
                            f += yLength;
                        }
                        __syncthreads();
                    }
                    __syncthreads();
                }
            }

            __syncthreads();
            devBuffer->rewind(zLength);
        }
#endif

        static inline void specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            /**
             * X holds data,
             * Y holds probabilities
             * Z will hold results
             */

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);

            // TODO: we probably might want to skip this sum, and state that probabilities array should be real probabilities, i.e. should sum to 1.0
            //T probSum = extraArguments[0];

            Nd4jLong yLength = shape::length(yShapeBuffer);
            Nd4jLong zLength = shape::length(zShapeBuffer);

            auto xEWS = shape::elementWiseStride(xShapeBuffer);
            auto yEWS = shape::elementWiseStride(yShapeBuffer);
            auto zEWS = shape::elementWiseStride(zShapeBuffer);

            int elementsPerThread = zLength / TAD_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            if (zEWS >= 1 && xEWS >= 1 && yEWS >= 1) {
#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                for (Nd4jLong e = 0; e < zLength; e++) {
                    T prob = buffer->relativeT<T>(e);
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
            } else {
                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];
                Nd4jLong zCoord[MAX_RANK];

                int xRank = shape::rank(xShapeBuffer);
                int yRank = shape::rank(yShapeBuffer);
                int zRank = shape::rank(zShapeBuffer);

                auto xShape = shape::shapeOf(xShapeBuffer);
                auto yShape = shape::shapeOf(yShapeBuffer);
                auto zShape = shape::shapeOf(zShapeBuffer);

                auto xStride = shape::stride(xShapeBuffer);
                auto yStride = shape::stride(yShapeBuffer);
                auto zStride = shape::stride(zShapeBuffer);


#pragma omp parallel for num_threads(_threads) if (_threads > 1) schedule(guided)
                for (Nd4jLong i = 0; i < zLength; i++) {
                    shape::ind2sub(zRank, zShape, i, zCoord);

                    auto zOffset2 = shape::getOffset(0, zShape, zStride, zCoord, zRank);

                    T prob = buffer->relativeT<T>(i);
                    T cumProb = (T) 0.0f;
                    for (Nd4jLong f = 0; f < yLength; f++) {
                        shape::ind2sub(yRank, yShape, i, yCoord);
                        auto yOffset2 = shape::getOffset(0, yShape, yStride, yCoord, yRank);

                        T relProb = y[yOffset2];
                        cumProb += relProb;

                        if (prob <= cumProb || f == yLength - 1) {
                            shape::ind2sub(xRank, xShape, f, xCoord);
                            Nd4jLong xOffset2 = shape::getOffset(0, xShape, xStride, xCoord, xRank);

                            z[zOffset2] = x[xOffset2];
                            f += yLength;
                        }
                    }
                }
            }

            // update rng state
            buffer->rewindH(zLength);
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
            __shared__ int zEWS;
            __shared__ int yEWS;
            __shared__ T mean;
            __shared__ T stddev;
            __shared__ int step;

            __shared__ T *tZ;

            __shared__ nd4j::random::RandomBuffer *buffer;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ nd4j::random::RandomBuffer *devBuffer;

            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                buffer = (nd4j::random::RandomBuffer *) shmem;
                cB = shmem;
                devBuffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
                dB = reinterpret_cast<unsigned char *> (state);

                tZ = (T *) (shmem + sizeof(nd4j::random::RandomBuffer));

                zLength = shape::length(zShapeBuffer);
                zEWS = shape::elementWiseStride(zShapeBuffer);
                yEWS = shape::elementWiseStride(yShapeBuffer);


                epsilon = (T) 1e-5;
                two_pi = (T) 2.0 * (T) 3.14159265358979323846;

                mean = extraArguments[0];
                stddev = extraArguments[1];

                step = (blockDim.x * gridDim.x);
            }
            __syncthreads();

            // using this loop instead of memcpy
            for (int e = threadIdx.x; e < sizeof(nd4j::random::RandomBuffer); e+= blockDim.x) {
                cB[e] = dB[e];
            }
            __syncthreads();

            Nd4jLong tid = blockIdx.x * blockDim.x + threadIdx.x;

            for (Nd4jLong e = tid; e < zLength; e += step) {
                // we need to get random values

                tZ[threadIdx.x] = buffer->relativeT<T>(e, epsilon, (T) 1.0f);

                // fix for "next rng value"
                if (e + 1 >= zLength && e % 2 == 0) {
                    tZ[threadIdx.x+1] = buffer->relativeT<T>(e+1, epsilon, (T) 1.0f);
                }

                T realMean = y == z ? mean : y[e * yEWS];

                __syncthreads();

                if (e % 2 == 0)
                    z[e *zEWS] =  (nd4j::math::nd4j_sqrt<T>((T) -2.0f * nd4j::math::nd4j_log<T>(tZ[threadIdx.x])) * nd4j::math::nd4j_cos<T>(two_pi * tZ[threadIdx.x+1])) * stddev + realMean;
                else
                    z[e *zEWS] =  (nd4j::math::nd4j_sqrt<T>((T) -2.0f * nd4j::math::nd4j_log<T>(tZ[threadIdx.x-1])) * nd4j::math::nd4j_sin<T>(two_pi * tZ[threadIdx.x])) * stddev + realMean;
                __syncthreads();
            }

            __syncthreads();
            devBuffer->rewind(zLength);
        }
#endif


        static inline void
        specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            const T two_pi = static_cast<T>(2.0f) * static_cast<T>(3.14159265358979323846);

            auto zLength = shape::length(zShapeBuffer);
            auto yEWS = shape::elementWiseStride(yShapeBuffer);
            auto zEWS = shape::elementWiseStride(zShapeBuffer);

            int elementsPerThread = zLength / TAD_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            int span = (zLength / _threads) + 8;

            // we're enforcing even chunks, since it's mandatory for this algorithm
            span -= span % 2;

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);

            T mean = extraArguments[0];
            T stddev = extraArguments[1];

#pragma omp parallel num_threads(_threads) if (_threads > 1) proc_bind(spread)
            {
                int tid = omp_get_thread_num();
                Nd4jLong start = span * tid;
                Nd4jLong end = span * (tid + 1);
                if (end > zLength) end = zLength;

                T z0, z1;
                T u0, u1;
                T lnU0;
                bool generated = false;

                for (Nd4jLong e = start; e < end; e++) {
                    if (!generated) {
                        /*
                         * Since box-muller transform expects non-zero u0 value, we'll just use rng with boundaries
                         */
                        u0 = buffer->relativeT<T>(e, (T) 1e-5f, (T) 1.0f);
                        u1 = buffer->relativeT<T>((e + 1), (T) 1e-5f, (T) 1.0f);
                        lnU0 = nd4j::math::nd4j_sqrt<T>((T) -2.0f * nd4j::math::nd4j_log<T>(u0));
                        z0 = lnU0 * nd4j::math::nd4j_cos<T>(two_pi * u1);
                        z1 = lnU0 * nd4j::math::nd4j_sin<T>(two_pi * u1);

                        generated = true;

                        T realMean = y == z ? mean : y[e * yEWS];

                        z[e * zEWS] = z0 * stddev + realMean;
                    } else {
                        T realMean = y == z ? mean : y[e * yEWS];

                        z[e * zEWS] = z1 * stddev + realMean;

                        generated = false;
                    }
                }
            }

            // update rng state
            buffer->rewindH(zLength);

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

            __shared__ nd4j::random::RandomBuffer *buffer;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ nd4j::random::RandomBuffer *devBuffer;
            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                buffer = reinterpret_cast<nd4j::random::RandomBuffer *>(shmem);
                cB = shmem;
                devBuffer = reinterpret_cast<nd4j::random::RandomBuffer *>(state);
                dB = reinterpret_cast<unsigned char *> (state);

                zLength = shape::length(zShapeBuffer);
                yEWS = shape::elementWiseStride(yShapeBuffer);
                zEWS = shape::elementWiseStride(zShapeBuffer);
            }
            __syncthreads();

            // using this loop instead of memcpy
            for (int e = threadIdx.x; e < sizeof(nd4j::random::RandomBuffer); e+= blockDim.x) {
                cB[e] = dB[e];
            }
            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            for (Nd4jLong e = tid; e < zLength; e += blockDim.x * gridDim.x) {
                int success = 0;
                for (int t = 1; t <= trials; t++) {
                    T randVal = buffer->relativeT<T>((e+1) * t);
                    if (y != z) {
                        // we're using external probs
                        prob = y[(t-1) * yEWS];
                    }

                    if (randVal < prob)
                        success++;
                }

                // we need this, to eliminate excessive code branching in runtime
                __syncthreads();

                // if trials is set to 0, effectively we just have successful memset
                z[e * zEWS] = static_cast<T>(success);
            }

            __syncthreads();
            if (trials > 0)
                devBuffer->rewind(zLength * trials);
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

            int span = (zLength / _threads) + 8;

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);

#pragma omp parallel num_threads(_threads) if (_threads > 1) proc_bind(spread)
            {
                int tid = omp_get_thread_num();
                Nd4jLong start = span * tid;
                Nd4jLong end = span * (tid + 1);
                if (end > zLength) end = zLength;

                T prob = extraArguments[1];

                for (Nd4jLong e = start; e < end; e++) {

                    int success = 0;
                    for (int t = 1; t <= trials; t++) {
                        T randVal = buffer->relativeT<T>((e+1) * t);
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
                buffer->rewindH(zLength * trials);
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

            __shared__ nd4j::random::RandomBuffer *buffer;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ nd4j::random::RandomBuffer *devBuffer;
            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                buffer = (nd4j::random::RandomBuffer *) shmem;
                cB = shmem;
                devBuffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
                dB = reinterpret_cast<unsigned char *> (state);

                zLength = shape::length(zShapeBuffer);
                yEWS = shape::elementWiseStride(yShapeBuffer);
                zEWS = shape::elementWiseStride(zShapeBuffer);
            }
            __syncthreads();

            // using this loop instead of memcpy
            for (int e = threadIdx.x; e < sizeof(nd4j::random::RandomBuffer); e+= blockDim.x) {
                cB[e] = dB[e];
            }
            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            for (Nd4jLong e = tid; e < zLength; e += blockDim.x * gridDim.x) {
                int success = 0;
                for (int t = 1; t <= trials; t++) {
                    T randVal = buffer->relativeT<T>((e+1) * t);
                    if (y != z) {
                        // we're using external probs
                        prob = y[e * yEWS];
                    }

                    if (randVal < prob)
                        success++;
                }

                // we need this, to eliminate excessive code branching in runtime
                __syncthreads();

                // if trials is set to 0, effectively we just have successful memset
                z[e * zEWS] = (T) success;
            }

            __syncthreads();
            if (trials > 0)
                devBuffer->rewind(zLength * trials);
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

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);

#pragma omp parallel num_threads(_threads) if (_threads > 1) proc_bind(spread)
            {
                int tid = omp_get_thread_num();
                Nd4jLong start = span * tid;
                Nd4jLong end = span * (tid + 1);
                if (end > zLength) end = zLength;

                T prob = extraArguments[1];

                for (Nd4jLong e = start; e < end; e++) {

                    int success = 0;
                    for (int t = 1; t <= trials; t++) {
                        T randVal = buffer->relativeT<T>((e+1) * t);
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
                buffer->rewindH(zLength * trials);
        }
    };
    
//////////////////////////////////////////////////////////////////////        
    // This Op produces random Gaussian values within [mean-2*stddev,mean+2*stddev]
    template<typename T>
    class TruncatedNormalDistribution {
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

            __shared__ nd4j::random::RandomBuffer *buffer;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ nd4j::random::RandomBuffer *devBuffer;

            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                buffer = reinterpret_cast<nd4j::random::RandomBuffer *>(shmem);
                cB = shmem;
                devBuffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
                dB = reinterpret_cast<unsigned char *> (state);

                tZ = reinterpret_cast<T *>(shmem + sizeof(nd4j::random::RandomBuffer));

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
            for (int e = threadIdx.x; e < sizeof(nd4j::random::RandomBuffer); e+= blockDim.x) {
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
                    u0 = buffer->relativeT<T>(e + generation0, epsilon, static_cast<T>(1.0f));
                    u1 = buffer->relativeT<T>(epm + generation0, epsilon, static_cast<T>(1.0f));

                    uT = nd4j::math::nd4j_sqrt<T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T>(u0));
                    uP = two_pi * u1;

                    z0 = uT * nd4j::math::nd4j_cos<T>(uP);
                    z1 = uT * nd4j::math::nd4j_sin<T>(uP);

                    result0 = z0 * stddev + realMean0;
                    result1 = z1 * stddev + realMean1;

                    generation0 += zLength;
                } while (ds < aRealMean0 + nd4j::math::nd4j_abs<T>(result0) || aRealMean1 + nd4j::math::nd4j_abs<T>(result1) > ds);

                z[e * zEWS] = result0;
                if((epm) < zLength)
                    z[epm * zEWS] = result1;
            }

            __syncthreads();
            devBuffer->rewind(zLength);
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

            int span = (middle / _threads) + 8;
            // we're enforcing even chunks, since it's mandatory for this algorithm
            span -= span % 2;

            nd4j::random::RandomBuffer *buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);

            T mean = extraArguments[0];
            T stddev = extraArguments[1];

#pragma omp parallel num_threads(_threads) if (_threads > 1) proc_bind(spread)
            {
                int tid = omp_get_thread_num();
                Nd4jLong start = span * tid; 
                Nd4jLong end = span * (tid + 1);
                if (end >  middle) {
                    end = middle;
                }
    
                T z0, z1;
                T u0, u1;
                T result0, result1, lnu0, lnu1;

                T ds = nd4j::math::nd4j_abs<T>(stddev) * (T) 2.0f;

                for (Nd4jLong e = start; e < end; e++) {
                   
                    /*
                    * Since box-muller transform expects non-zero u0 value, we'll just use rng with boundaries
                    */
                    Nd4jLong generation0 = 0;
                    auto epm = e + middle;
                    T realMean0 = y == z ? mean : y[e * yEWS];
                    T realMean1 = y == z ? mean : y[epm * yEWS];
                    T aRealMean0 = nd4j::math::nd4j_abs<T>(realMean0);
                    T aRealMean1 = nd4j::math::nd4j_abs<T>(realMean1);
                    do {
                        u0 = buffer->relativeT<T>(e + generation0, static_cast<T>(1e-6f), static_cast<T>(1.0f));
                        u1 = buffer->relativeT<T>((epm + generation0), static_cast<T>(1e-6f), static_cast<T>(1.0f));
                        lnu0 = nd4j::math::nd4j_sqrt<T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T>(u0));
                        lnu1 = two_pi * u1;
                        z0 = lnu0 * nd4j::math::nd4j_cos<T>(lnu1);
                        z1 = lnu0 * nd4j::math::nd4j_sin<T>(lnu1);

                        result0 = z0 * stddev + realMean0;
                        result1 = z1 * stddev + realMean1;
                        generation0 += zLength;
                    } while (aRealMean0 + nd4j::math::nd4j_abs<T>(result0) > ds || aRealMean1 + nd4j::math::nd4j_abs<T>(result1) > ds);

                    z[e*zEWS] = result0;
                    if(epm < zLength)
                        z[epm * zEWS] = result1;
                }                            
            }
            // update rng state
            buffer->rewindH(zLength);

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

            __shared__ nd4j::random::RandomBuffer *buffer;
            __shared__ unsigned char *cB;
            __shared__ unsigned char *dB;
            __shared__ nd4j::random::RandomBuffer *devBuffer;

            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                buffer = reinterpret_cast<nd4j::random::RandomBuffer *>(shmem);
                cB = shmem;
                devBuffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);
                dB = reinterpret_cast<unsigned char *> (state);

                tZ = reinterpret_cast<T*>(shmem + sizeof(nd4j::random::RandomBuffer));

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
            for (int e = threadIdx.x; e < sizeof(nd4j::random::RandomBuffer); e+= blockDim.x) {
                cB[e] = dB[e];
            }
            __syncthreads();

            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            for (Nd4jLong e = tid; e < zLength; e += step) {
                // we need to get random values

                tZ[threadIdx.x] = buffer->relativeT<T>(e, epsilon, static_cast<T>(1.0f));

                // fix for "next rng value"
                if (e + 1 >= zLength && e % 2 == 0) {
                    tZ[threadIdx.x+1] = buffer->relativeT<T>(e+1, epsilon, static_cast<T>(1.0f));
                }

                T realMean = y == z ? mean : y[e * yEWS];

                __syncthreads();

                if (e % 2 == 0)
                    z[e *zEWS] =  nd4j::math::nd4j_exp<T>((nd4j::math::nd4j_sqrt<T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T>(tZ[threadIdx.x])) * nd4j::math::nd4j_cos<T>(two_pi * tZ[threadIdx.x+1])) * stddev + realMean);
                else
                    z[e *zEWS] =  nd4j::math::nd4j_exp<T>((nd4j::math::nd4j_sqrt<T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T>(tZ[threadIdx.x-1])) * nd4j::math::nd4j_sin<T>(two_pi * tZ[threadIdx.x])) * stddev + realMean);
                __syncthreads();
            }

            __syncthreads();
            devBuffer->rewind(zLength);
        }
#endif

        static inline void
        specialOp(Nd4jPointer state, T *x, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *z, Nd4jLong *zShapeBuffer, T *extraArguments) {
            const T two_pi = static_cast<T>(2.0f) * static_cast<T>(3.14159265358979323846);

            Nd4jLong zLength = shape::length(zShapeBuffer);
            auto yEWS = shape::elementWiseStride(yShapeBuffer);
            auto zEWS = shape::elementWiseStride(zShapeBuffer);

            int elementsPerThread = zLength / TAD_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            int span = (zLength / _threads) + 8;

            // we're enforcing even chunks, since it's mandatory for this algorithm
            span -= span % 2;

            auto buffer = reinterpret_cast<nd4j::random::RandomBuffer *> (state);

            T mean = extraArguments[0];
            T stddev = extraArguments[1];

#pragma omp parallel num_threads(_threads) if (_threads > 1) proc_bind(spread)
            {
                int tid = omp_get_thread_num();
                Nd4jLong start = span * tid;
                Nd4jLong end = span * (tid + 1);
                if (end > zLength) end = zLength;

                T z0, z1;
                T u0, u1;
                T lnU0;
                bool generated = false;

                for (Nd4jLong e = start; e < end; e++) {
                    if (!generated) {
                        /*
                         * Since box-muller transform expects non-zero u0 value, we'll just use rng with boundaries
                         */
                        u0 = buffer->relativeT<T>(e, static_cast<T>(1e-5f), static_cast<T>(1.0f));
                        u1 = buffer->relativeT<T>((e + 1), static_cast<T>(1e-5f), static_cast<T>(1.0f));
                        lnU0 = nd4j::math::nd4j_sqrt<T>(static_cast<T>(-2.0f) * nd4j::math::nd4j_log<T>(u0));
                        z0 = lnU0 * nd4j::math::nd4j_cos<T>(two_pi * u1);
                        z1 = lnU0 * nd4j::math::nd4j_sin<T>(two_pi * u1);

                        generated = true;

                        T realMean = y == z ? mean : y[e * yEWS];

                        z[e * zEWS] = nd4j::math::nd4j_exp<T>(z0 * stddev + realMean);
                    } else {
                        T realMean = y == z ? mean : y[e * yEWS];

                        z[e * zEWS] = nd4j::math::nd4j_exp<T>(z1 * stddev + realMean);

                        generated = false;
                    }
                }
            }

            // update rng state
            buffer->rewindH(zLength);

        }
    };


}

#endif //LIBND4J_SPECIAL_RANDOM_OPS_H
