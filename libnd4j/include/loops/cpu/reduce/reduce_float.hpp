#include <helpers/ConstantTadHelper.h>
#include <helpers/Loops.h>
#include <helpers/OmpLaunchHelper.h>
#include <helpers/ShapeBuilders.h>
#include <loops/legacy_ops.h>
#include <loops/reduce_float.h>
#include <system/op_boilerplate.h>
#include <types/types.h>
#include <algorithm>

using namespace simdOps;

namespace functions {
namespace reduce {

// =============================================================================
// TYPE-SAFE UTILITIES FOR ALL NUMERIC TYPES INCLUDING FLOAT16
// =============================================================================

namespace SafeTypeUtils {

/**
 * @brief Type-safe array initialization that works with float16 and all other types
 */
template<typename T>
SD_INLINE void initializeArray(T* array, size_t count) {
  if constexpr (std::is_arithmetic_v<T>) {
    // For arithmetic types including float16, use loop initialization
    for (size_t i = 0; i < count; i++) {
      array[i] = static_cast<T>(0);
    }
  } else {
    // For non-arithmetic types, use default initialization
    std::fill_n(array, count, T{});
  }
}

/**
 * @brief Safe type conversion for mixed-type operations
 */
template<typename From, typename To>
SD_INLINE constexpr To safeCast(const From& value) {
  return static_cast<To>(value);
}

/**
 * @brief Convert parameter arrays between types safely
 */
template<typename SourceType, typename TargetType>
SD_INLINE void convertParams(const SourceType* source, TargetType* target, size_t count = 8) {
  if (source && target) {
    for (size_t i = 0; i < count; i++) {
      target[i] = safeCast<SourceType, TargetType>(source[i]);
    }
  }
}

/**
 * @brief Determine appropriate parameter type for mixed operations
 * For float16, use float for better compatibility; otherwise use Z type
 */
template<typename X, typename Z>
struct CompatibleParamType {
  using type = typename std::conditional_t<
      std::is_same_v<Z, float16> || std::is_same_v<Z, bfloat16>,
      float,  // Use float for half-precision types
      Z       // Use Z for all other types
      >;
};

} // namespace SafeTypeUtils

// =============================================================================
// REDUCE FLOAT FUNCTION IMPLEMENTATION WITH FLOAT16 SUPPORT
// =============================================================================

template <typename X, typename Z>
template <typename OpType>
void SD_HOST ReduceFloatFunction<X, Z>::execScalar(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams,
                                                   void *vz, const sd::LongType *zShapeInfo) {
  auto x = reinterpret_cast<const X *>(vx);
  auto z = reinterpret_cast<Z *>(vz);

  // Convert to Z* for consistency with macro expectations
  Z *extraParams = nullptr;
  Z convertedParams[8];

  if (vextraParams != nullptr) {
    if constexpr (std::is_same_v<Z, X>) {
      extraParams = reinterpret_cast<Z*>(vextraParams);
    } else {
      // Convert parameters to Z type
      auto originalParams = reinterpret_cast<X*>(vextraParams);
      for (int i = 0; i < 8; ++i) {
        convertedParams[i] = static_cast<Z>(originalParams[i]);
      }
      extraParams = convertedParams;
    }
  }

  const auto length = shape::length(xShapeInfo);

  if (shape::isEmptyConst(xShapeInfo)) {
    z[0] = static_cast<Z>(OpType::startingValue(x));
    return;
  }

  if (sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY) {
    if (sd::ArrayOptions::arrayType(zShapeInfo) == sd::ArrayType::EMPTY) return;
    const auto startingVal = static_cast<Z>(OpType::startingValue(x));

    for (sd::LongType i = 0; i < length; i++) {
      z[i] = startingVal;
    }
    return;
  }

  auto startingValue = static_cast<typename OpType::InterType>(OpType::startingValue(x));
  int maxThreads = sd::math::sd_min<int>(64, sd::Environment::getInstance().maxThreads());
  typename OpType::InterType intermediate[64];

  PRAGMA_OMP_SIMD
  for (auto e = 0; e < maxThreads; e++) {
    intermediate[e] = startingValue;
  }

  sd::LongType xRank = shape::rank(xShapeInfo);
  sd::LongType* xShape = shape::shapeOf(xShapeInfo);
  sd::LongType* xStride = shape::stride(xShapeInfo);

  if(shape::isViewConst(xShapeInfo)) {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        sd::LongType coords[SD_MAX_RANK];
        INDEX2COORDS(i, xRank, xShape, coords);
        sd::LongType indexOffset;
        COORDS2INDEX(xRank, xStride, coords, indexOffset);

        auto opResult = OpType::op(x[indexOffset], extraParams);
        intermediate[thread_id] = OpType::update(
            intermediate[thread_id],
            opResult,
            extraParams
        );
      }
    };
    maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);

    PRAGMA_OMP_SIMD
    for (int e = 1; e < maxThreads; e++) {
      intermediate[0] = OpType::merge(intermediate[0], intermediate[e], extraParams);
    }

    z[0] = OpType::postProcess(intermediate[0], length, extraParams);
  } else {
    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        auto opResult = OpType::op(x[i], extraParams);
        intermediate[thread_id] = OpType::update(
            intermediate[thread_id],
            opResult,
            extraParams
        );
      }
    };
    maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);

    PRAGMA_OMP_SIMD
    for (int e = 1; e < maxThreads; e++) {
      intermediate[0] = OpType::merge(intermediate[0], intermediate[e], extraParams);
    }

    z[0] = OpType::postProcess(intermediate[0], length, extraParams);
  }
}

template <typename X, typename Z>
template <typename OpType>
Z SD_HOST ReduceFloatFunction<X, Z>::execScalar(const void *vx, const sd::LongType *xShapeInfo, void *vextraParams) {
  auto x = reinterpret_cast<const X *>(vx);

  // Convert to Z* for compatibility with OpType::op
  Z *extraParams = nullptr;
  Z convertedParams[8];

  if (vextraParams != nullptr) {
    if constexpr (std::is_same_v<Z, X>) {
      extraParams = reinterpret_cast<Z*>(vextraParams);
    } else {
      // Convert the parameters to Z type
      auto originalParams = reinterpret_cast<X*>(vextraParams);
      for (int i = 0; i < 8; ++i) {
        convertedParams[i] = static_cast<Z>(originalParams[i]);
      }
      extraParams = convertedParams;
    }
  }

  const sd::LongType length = shape::length(xShapeInfo);
  auto startingValue = static_cast<typename OpType::InterType>(OpType::startingValue(x));

  sd::LongType xRank = shape::rank(xShapeInfo);
  sd::LongType *xShape = shape::shapeOf(xShapeInfo);
  sd::LongType *xStride = shape::stride(xShapeInfo);

  for (sd::LongType i = 0; i < length; i++) {
    sd::LongType coords[SD_MAX_RANK];
    INDEX2COORDS(i, xRank, xShape, coords);
    sd::LongType offset;
    COORDS2INDEX(xRank, xStride, coords, offset);

    auto opResult = OpType::op(x[offset], extraParams);
    startingValue = OpType::update(startingValue, opResult, extraParams);
  }

  return OpType::postProcess(startingValue, length, extraParams);
}
template <typename X, typename Z>
template <typename OpType>
void SD_HOST ReduceFloatFunction<X, Z>::exec(sd::memory::Workspace *workspace, const void *vx,
                                             const sd::LongType *xShapeInfo, void *vextraParams, void *vz,
                                             const sd::LongType *zShapeInfo, const long long int *dims) {
  const X *x = reinterpret_cast<const X *>(vx);
  Z *z = reinterpret_cast<Z *>(vz);

  // CRITICAL FIX: Type-safe parameter handling for all numeric types
  using CompatibleParamType = typename SafeTypeUtils::CompatibleParamType<X, Z>::type;
  CompatibleParamType *compatibleExtraParams = nullptr;
  CompatibleParamType convertedParams[8];
  SafeTypeUtils::initializeArray(convertedParams, 8);

  if (vextraParams != nullptr) {
    if constexpr (std::is_same_v<X, CompatibleParamType>) {
      compatibleExtraParams = reinterpret_cast<CompatibleParamType*>(vextraParams);
    } else {
      SafeTypeUtils::convertParams(reinterpret_cast<X*>(vextraParams), convertedParams, 8);
      compatibleExtraParams = convertedParams;
    }
  }

  const int xRank = shape::rank(xShapeInfo);
  const int zRank = shape::rank(zShapeInfo);

  if (sd::ArrayOptions::arrayType(xShapeInfo) == sd::ArrayType::EMPTY) {
    const auto startingVal = std::is_same<OpType, simdOps::Mean<X, Z>>::value
                                 ? sd::DataTypeUtils::nanOrZero<Z>()
                                 : SafeTypeUtils::safeCast<X, Z>(OpType::startingValue(x));
    const auto zLen = shape::length(zShapeInfo);
    if (z != nullptr)
      for (sd::LongType i = 0; i < zLen; i++) z[i] = startingVal;
    return;
  }

  if (shape::length(zShapeInfo) == 1) {
    z[0] = execScalar<OpType>(x, xShapeInfo, compatibleExtraParams);
    return;
  }

  if (OpType::requiresSpecialAccumulation) {
    // FIXED: Handle execSpecial with flexible parameter types
    // The enhanced macro provides template overloads that accept any arithmetic type
        
    if constexpr (std::is_same_v<CompatibleParamType, sd::LongType>) {
      // Direct call for sd::LongType parameters
      OpType::execSpecial(x, xShapeInfo, compatibleExtraParams, z, zShapeInfo,
                          const_cast<sd::LongType *>(dims) + zRank, xRank - zRank,
                          nullptr, nullptr);
    } else {
      // Convert to sd::LongType for operations that specifically need it
      sd::LongType longExtraParams[8];
      SafeTypeUtils::initializeArray(longExtraParams, 8);
            
      if (compatibleExtraParams != nullptr) {
        SafeTypeUtils::convertParams(compatibleExtraParams, longExtraParams, 8);
      }
            
      // Use template overload that accepts sd::LongType*
      OpType::execSpecial(x, xShapeInfo, longExtraParams, z, zShapeInfo,
                          const_cast<sd::LongType *>(dims) + zRank, xRank - zRank,
                          nullptr, nullptr);
    }
    return;
  }

#ifdef SD_LOOPS_INLINED
  sd::ReductionLoops<X, Z, CompatibleParamType>::template loopReduce<OpType>(workspace, x, xShapeInfo, z, zShapeInfo, dims, compatibleExtraParams);
#else
  sd::ReductionFloatLoops<X, Z>::template innerloopReduce<OpType>(workspace, x, xShapeInfo, z, zShapeInfo, dims, compatibleExtraParams);
#endif
}


template <typename X, typename Y>
Y ReduceFloatFunction<X, Y>::execScalar(const int opNum, const void *x, const sd::LongType *xShapeInfo,
                                        void *extraParams) {
  RETURNING_DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams), REDUCE_FLOAT_OPS);
}

template <typename X, typename Y>
void ReduceFloatFunction<X, Y>::execScalar(const int opNum, const void *x, const sd::LongType *xShapeInfo,
                                           void *extraParams, void *z, const sd::LongType *zShapeInfo) {
  DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo), REDUCE_FLOAT_OPS);
}

template <typename X, typename Z>
template <typename OpType>
void SD_HOST ReduceFloatFunction<X, Z>::exec(const void *x, const sd::LongType *xShapeInfo, void *extraParams,
                                             void *vresult, const sd::LongType *resultShapeInfo) {
  auto z = reinterpret_cast<Z *>(vresult);
  z[0] = execScalar<OpType>(x, xShapeInfo, extraParams);
}

template <typename X, typename Z>
template <typename OpType>
Z SD_HOST ReduceFloatFunction<X, Z>::execScalar(const void *vx, sd::LongType xEws, sd::LongType length,
                                                void *vextraParams) {
  auto x = reinterpret_cast<const X *>(vx);

  using CompatibleParamType = typename SafeTypeUtils::CompatibleParamType<X, Z>::type;
  CompatibleParamType *compatibleExtraParams = nullptr;
  CompatibleParamType convertedParams[8];
  SafeTypeUtils::initializeArray(convertedParams, 8);

  if (vextraParams != nullptr) {
    if constexpr (std::is_same_v<X, CompatibleParamType>) {
      compatibleExtraParams = reinterpret_cast<CompatibleParamType*>(vextraParams);
    } else {
      SafeTypeUtils::convertParams(reinterpret_cast<X*>(vextraParams), convertedParams, 8);
      compatibleExtraParams = convertedParams;
    }
  }

  int maxThreads = sd::math::sd_min<int>(64, sd::Environment::getInstance().maxThreads());
  using InterType = typename OpType::InterType;
  InterType intermediate[64];

  PRAGMA_OMP_SIMD
  for (auto e = 0; e < maxThreads; e++) {
    intermediate[e] = SafeTypeUtils::safeCast<X, InterType>(OpType::startingValue(x));
  }

  auto func = PRAGMA_THREADS_FOR {
    if (xEws == 1) {
      for (auto i = start; i < stop; i++) {
        auto opResult = OpType::op(x[i], compatibleExtraParams);
        intermediate[thread_id] = OpType::update(
            intermediate[thread_id],
            SafeTypeUtils::safeCast<decltype(opResult), InterType>(opResult),
            compatibleExtraParams
        );
      }
    } else {
      for (auto i = start; i < stop; i++) {
        auto opResult = OpType::op(x[i * xEws], compatibleExtraParams);
        intermediate[thread_id] = OpType::update(
            intermediate[thread_id],
            SafeTypeUtils::safeCast<decltype(opResult), InterType>(opResult),
            compatibleExtraParams
        );
      }
    }
  };

  maxThreads = samediff::Threads::parallel_for(func, 0, length, 1, maxThreads);

  for (int e = 1; e < maxThreads; e++)
    intermediate[0] = OpType::update(intermediate[0], intermediate[e], compatibleExtraParams);

  return SafeTypeUtils::safeCast<InterType, Z>(OpType::postProcess(intermediate[0], length, compatibleExtraParams));
}

template <typename X, typename Y>
void ReduceFloatFunction<X, Y>::exec(int opNum, sd::memory::Workspace *workspace, const void *vx,
                                     const sd::LongType *xShapeInfo, void *vextraParams, void *vz,
                                     const sd::LongType *zShapeInfo, const long long int *dims) {
  DISPATCH_BY_OPNUM_TT(exec, PARAMS(workspace, vx, xShapeInfo, vextraParams, vz, zShapeInfo, dims), REDUCE_FLOAT_OPS);
}


}  // namespace reduce
}  // namespace functions