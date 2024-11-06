#ifndef LIBND4J_SPECIAL_RANDOM_OPS_H
#define LIBND4J_SPECIAL_RANDOM_OPS_H
#include <execution/Threads.h>
#include <graph/RandomGenerator.h>
#include <helpers/shape.h>
#include <ops/random_ops.h>
#include <ops/specials_cuda.h>

namespace randomOps {

//////////////////////////////////////////////////////////////////////
template <typename T>
class Choice {
 public:
  method_idx method_X method_XY

      static const bool requiresSpecial = true;

#ifdef __CUDACC__
  static SD_INLINE SD_DEVICE void specialOpCuda(sd::Pointer state, T const *x, sd::LongType const *xShapeBuffer,
                                                T const *y, sd::LongType const *yShapeBuffer, T *z,
                                                sd::LongType const *zShapeBuffer, T *extraArguments) {
    // ... (CUDA implementation remains unchanged)
  }
#endif

  static inline void specialOp(sd::Pointer state, const T *x, const sd::LongType *xShapeBuffer, const T *y,
                               const sd::LongType *yShapeBuffer, T *z, const sd::LongType *zShapeBuffer,
                               T *extraArguments) {
    sd::LongType zLength = shape::length(zShapeBuffer);
    sd::LongType yLength = shape::length(yShapeBuffer);

    int elementsPerThread = zLength / TAD_THRESHOLD;
    int _threads = sd::math::sd_max<int>(1, elementsPerThread);
    _threads = sd::math::sd_min<int>(_threads, sd::Environment::getInstance().maxThreads());

    sd::graph::RandomGenerator *rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);
    auto func = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        sd::LongType coords[SD_MAX_RANK];
        shape::index2coords(e, zShapeBuffer, coords);
        auto zOffset = shape::getOffset(zShapeBuffer, coords);
        
        T prob = rng->relativeT<T>(e);
        T cumProb = (T)0.0f;
        for (sd::LongType f = 0; f < yLength; f++) {
          sd::LongType yCoords[SD_MAX_RANK];
          shape::index2coords(f, yShapeBuffer, yCoords);
          auto yOffset = shape::getOffset(yShapeBuffer, yCoords);
          T relProb = y[yOffset];
          cumProb += relProb;

          if (prob <= cumProb || f == yLength - 1) {
            sd::LongType xCoords[SD_MAX_RANK];
            shape::index2coords(f, xShapeBuffer, xCoords);
            auto xOffset = shape::getOffset(xShapeBuffer, xCoords);
            z[zOffset] = x[xOffset];
            break;
          }
        }
      }
    };

    samediff::Threads::parallel_for(func, 0, zLength, 1, _threads);
  }
};

//////////////////////////////////////////////////////////////////////
template <typename T>
class GaussianDistribution {
 public:
  method_XY method_X method_idx

      static const bool requiresSpecial = true;

#ifdef __CUDACC__
  static SD_INLINE SD_DEVICE void specialOpCuda(sd::Pointer state, T const *x, sd::LongType const *xShapeBuffer,
                                                T const *y, sd::LongType const *yShapeBuffer, T *z,
                                                sd::LongType const *zShapeBuffer, T *extraArguments) {
    // ... (CUDA implementation remains unchanged)
  }
#endif

  static inline void specialOp(sd::Pointer state, const T *x, const sd::LongType *xShapeBuffer, const T *y,
                               const sd::LongType *yShapeBuffer, T *z, const sd::LongType *zShapeBuffer,
                               T *extraArguments) {
    const T two_pi = static_cast<T>(2.0f) * static_cast<T>(3.14159265358979323846);

    sd::LongType zLength = shape::length(zShapeBuffer);
    auto middle = zLength % 2 + zLength / 2;

    int elementsPerThread = middle / TAD_THRESHOLD;
    int _threads = sd::math::sd_max<int>(1, elementsPerThread);
    _threads = sd::math::sd_min<int>(_threads, sd::Environment::getInstance().maxThreads());

    sd::graph::RandomGenerator *rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);
    const T mean = extraArguments[0];
    const T stddev = extraArguments[1];

    const T epsilon = static_cast<T>(1e-5);

    auto func = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        sd::LongType coords[SD_MAX_RANK];
        shape::index2coords(e, zShapeBuffer, coords);
        auto zOffset = shape::getOffset(zShapeBuffer, coords);
        auto epm = e + middle;

        // we need to get random values
        T r0 = rng->relativeT<T>(e, epsilon, static_cast<T>(1.0f));
        T r1 = rng->relativeT<T>(epm, epsilon, static_cast<T>(1.0f));

        auto yOffset = shape::getOffset(yShapeBuffer, coords);
        T realMean0 = y == z ? mean : y[yOffset];

        z[zOffset] = (sd::math::sd_sqrt<T, T>(static_cast<T>(-2.0f) * sd::math::sd_log<T, T>(r0)) *
                      sd::math::sd_cos<T, T>(two_pi * r1)) * stddev + realMean0;

        if (epm < zLength) {
          shape::index2coords(epm, zShapeBuffer, coords);
          zOffset = shape::getOffset(zShapeBuffer, coords);
          yOffset = shape::getOffset(yShapeBuffer, coords);
          T realMean1 = y == z ? mean : y[yOffset];
          z[zOffset] = (sd::math::sd_sqrt<T, T>(static_cast<T>(-2.0f) * sd::math::sd_log<T, T>(r0)) *
                        sd::math::sd_sin<T, T>(two_pi * r1)) * stddev + realMean1;
        }
      }
    };

    samediff::Threads::parallel_for(func, 0, middle, 1, _threads);
  }
};

//////////////////////////////////////////////////////////////////////
template <typename T>
class BinomialDistribution {
 public:
  method_XY method_X method_idx

      static const bool requiresSpecial = true;

#ifdef __CUDACC__
  static SD_INLINE SD_DEVICE void specialOpCuda(sd::Pointer state, T const *x, sd::LongType const *xShapeBuffer,
                                                T const *y, sd::LongType const *yShapeBuffer, T *z,
                                                sd::LongType const *zShapeBuffer, T *extraArguments) {
    // ... (CUDA implementation remains unchanged)
  }
#endif

  static inline void specialOp(sd::Pointer state, const T *x, const sd::LongType *xShapeBuffer, const T *y,
                               const sd::LongType *yShapeBuffer, T *z, const sd::LongType *zShapeBuffer,
                               T *extraArguments) {
    int trials = (int)extraArguments[0];

    sd::LongType zLength = shape::length(zShapeBuffer);

    int elementsPerThread = zLength / TAD_THRESHOLD;
    int _threads = sd::math::sd_max<int>(1, elementsPerThread);
    _threads = sd::math::sd_min<int>(_threads, sd::Environment::getInstance().maxThreads());

    T prob = extraArguments[1];

    sd::graph::RandomGenerator *rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);
    auto func = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        sd::LongType coords[SD_MAX_RANK];
        shape::index2coords(e, zShapeBuffer, coords);
        auto zOffset = shape::getOffset(zShapeBuffer, coords);
        
        int success = 0;
        for (int t = 1; t <= trials; t++) {
          T randVal = rng->relativeT<T>((e + 1) * t);
          if (y != z) {
            // we're using external probs
            auto yOffset = shape::getOffset(yShapeBuffer, coords);
            prob = y[yOffset];
          }

          if (randVal < prob) success++;
        }

        // if trials is set to 0, effectively we just have successful memset
        z[zOffset] = static_cast<T>(success);
      }
    };

    samediff::Threads::parallel_for(func, 0, zLength, 1, _threads);
  }
};

//////////////////////////////////////////////////////////////////////
template <typename T>
class BinomialDistributionEx {
 public:
  method_XY method_X method_idx

      static const bool requiresSpecial = true;

#ifdef __CUDACC__
  static SD_INLINE SD_DEVICE void specialOpCuda(sd::Pointer state, T const *x, sd::LongType const *xShapeBuffer,
                                                T const *y, sd::LongType const *yShapeBuffer, T *z,
                                                sd::LongType const *zShapeBuffer, T *extraArguments) {
    // ... (CUDA implementation remains unchanged)
  }
#endif

  static inline void specialOp(sd::Pointer state, const T *x, const sd::LongType *xShapeBuffer, const T *y,
                               const sd::LongType *yShapeBuffer, T *z, const sd::LongType *zShapeBuffer,
                               T *extraArguments) {
    int trials = (int)extraArguments[0];

    sd::LongType zLength = shape::length(zShapeBuffer);

    int elementsPerThread = zLength / TAD_THRESHOLD;
    int _threads = sd::math::sd_max<int>(1, elementsPerThread);
    _threads = sd::math::sd_min<int>(_threads, sd::Environment::getInstance().maxThreads());

    T prob = extraArguments[1];

    auto rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);
    auto func = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        sd::LongType coords[SD_MAX_RANK];
        shape::index2coords(e, zShapeBuffer, coords);
        auto zOffset = shape::getOffset(zShapeBuffer, coords);
        
        int success = 0;
        for (int t = 1; t <= trials; t++) {
          T randVal = rng->relativeT<T>((e + 1) * t);
          if (y != z) {
            // we're using external probs
            auto yOffset = shape::getOffset(yShapeBuffer, coords);
            prob = y[yOffset];
          }

          if (randVal < prob) success++;
        }

        // if trials is set to 0, effectively we just have successful memset
        z[zOffset] = static_cast<T>(success);
      }
    };

    samediff::Threads::parallel_for(func, 0, zLength, 1, _threads);
  }
};

//////////////////////////////////////////////////////////////////////
template <typename T>
class TruncatedNormalDistribution {
 private:
  static SD_INLINE SD_HOST_DEVICE T step(sd::graph::RandomGenerator *rng, T mean, T stddev, sd::LongType e,
                                         sd::LongType middle, T &z) {
    auto epm = e + middle;
    const T two_pi = static_cast<T>(2.0f) * static_cast<T>(3.14159265358979323846);
    const T epsilon = static_cast<T>(1.e-5f);
    // we need to get random values
    T r0 = rng->relativeT<T>(e, epsilon, static_cast<T>(1.0f));
    T r1 = rng->relativeT<T>(epm, epsilon, static_cast<T>(1.0f));

    T realMean0 = mean;

    auto z0 = (sd::math::sd_sqrt<T, T>(static_cast<T>(-2.0f) * sd::math::sd_log<T, T>(r0)) *
               sd::math::sd_cos<T, T>(two_pi * r1)) *
                  stddev +
              realMean0;
    z = z0;
    if (epm < middle) {
      T realMean1 = mean;
      auto z1 = (sd::math::sd_sqrt<T, T>(static_cast<T>(-2.0f) * sd::math::sd_log<T, T>(r0)) *
                 sd::math::sd_sin<T, T>(two_pi * r1)) *
                    stddev +
                realMean1;
      z = z1;
    }
    return z;
  }

 public:
  method_XY method_X method_idx

      static const bool requiresSpecial = true;

#ifdef __CUDACC__
  static SD_INLINE SD_DEVICE void specialOpCuda(sd::Pointer state, T const *x, sd::LongType const *xShapeBuffer,
                                                T const *y, sd::LongType const *yShapeBuffer, T *z,
                                                sd::LongType const *zShapeBuffer, T *extraArguments) {
    // ... (CUDA implementation remains unchanged)
  }
#endif

  static inline void specialOp(sd::Pointer state, const T *x, const sd::LongType *xShapeBuffer, const T *y,
                               const sd::LongType *yShapeBuffer, T *z, const sd::LongType *zShapeBuffer,
                               T *extraArguments) {
    GaussianDistribution<T>::specialOp(state, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments);
    sd::LongType zLength = shape::length(zShapeBuffer);
    auto rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);
    T mean = extraArguments[0];
    T stddev = extraArguments[1];
    T ds = sd::math::sd_abs<T,T>(stddev) * (T)2.0f;
    sd::LongType middle = zLength / 2 + (zLength % 2);
    int elementsPerThread = middle / TAD_THRESHOLD;
    int _threads = sd::math::sd_max<int>(1, elementsPerThread);
    _threads = sd::math::sd_min<int>(_threads, sd::Environment::getInstance().maxThreads());

    const T epsilon = static_cast<T>(1e-5);

    auto func = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        sd::LongType coords[SD_MAX_RANK];
        shape::index2coords(e, zShapeBuffer, coords);
        auto zOffset = shape::getOffset(zShapeBuffer, coords);

        if (z[zOffset] > mean + ds || z[zOffset] < mean - ds) {
          z[zOffset] = step(rng, mean, stddev, e, middle, z[zOffset]);

          if (z[zOffset] > mean + ds || z[zOffset] < mean - ds) z[zOffset] = mean + sd::DataTypeUtils::min_positive<T>();
        }
      }
    };

    samediff::Threads::parallel_for(func, 0, zLength, 1, _threads);
  }
};

//////////////////////////////////////////////////////////////////////
template <typename T>
class LogNormalDistribution {
 public:
  method_XY method_X method_idx

      static const bool requiresSpecial = true;

#ifdef __CUDACC__
  static SD_INLINE SD_DEVICE void specialOpCuda(sd::Pointer state, T const *x, sd::LongType const *xShapeBuffer,
                                                T const *y, sd::LongType const *yShapeBuffer, T *z,
                                                sd::LongType const *zShapeBuffer, T *extraArguments) {
    // ... (CUDA implementation remains unchanged)
  }
#endif

  static inline void specialOp(sd::Pointer state, const T *x, const sd::LongType *xShapeBuffer, const T *y,
                               const sd::LongType *yShapeBuffer, T *z, const sd::LongType *zShapeBuffer,
                               T *extraArguments) {
    const T two_pi = static_cast<T>(2.0f) * static_cast<T>(3.14159265358979323846);

    sd::LongType zLength = shape::length(zShapeBuffer);
    auto middle = zLength % 2 == 0 ? zLength / 2 : zLength / 2 + 1;

    int elementsPerThread = middle / TAD_THRESHOLD;
    int _threads = sd::math::sd_max<int>(1, elementsPerThread);
    _threads = sd::math::sd_min<int>(_threads, sd::Environment::getInstance().maxThreads());

    auto rng = reinterpret_cast<sd::graph::RandomGenerator *>(state);

    const T mean = extraArguments[0];
    const T stddev = extraArguments[1];
    const T epsilon = static_cast<T>(1e-5);

    auto func = PRAGMA_THREADS_FOR {
      for (auto e = start; e < stop; e++) {
        sd::LongType coords[SD_MAX_RANK];
        shape::index2coords(e, zShapeBuffer, coords);
        auto zOffset = shape::getOffset(zShapeBuffer, coords);
        auto epm = e + middle;

        // we need to get random values
        T r0 = rng->relativeT<T>(e, epsilon, static_cast<T>(1.0f));
        T r1 = rng->relativeT<T>(epm, epsilon, static_cast<T>(1.0f));

        auto yOffset = shape::getOffset(yShapeBuffer, coords);
        T realMean = y == z ? mean : y[yOffset];

        z[zOffset] =
            sd::math::sd_exp<T, T>((sd::math::sd_sqrt<T, T>(static_cast<T>(-2.0f) * sd::math::sd_log<T, T>(r0)) *
                                    sd::math::sd_cos<T, T>(two_pi * r1)) *
                                       stddev +
                                   realMean);

        if (epm < zLength) {
          shape::index2coords(epm, zShapeBuffer, coords);
          zOffset = shape::getOffset(zShapeBuffer, coords);
          yOffset = shape::getOffset(yShapeBuffer, coords);
          realMean = y == z ? mean : y[yOffset];
          z[zOffset] =
              sd::math::sd_exp<T, T>((sd::math::sd_sqrt<T, T>(static_cast<T>(-2.0f) * sd::math::sd_log<T, T>(r0)) *
                                      sd::math::sd_sin<T, T>(two_pi * r1)) *
                                         stddev +
                                     realMean);
        }
      }
    };

    samediff::Threads::parallel_for(func, 0, middle, 1, _threads);
  }
};

}  // namespace randomOps

#endif  // LIBND4J_SPECIAL_RANDOM_OPS_H