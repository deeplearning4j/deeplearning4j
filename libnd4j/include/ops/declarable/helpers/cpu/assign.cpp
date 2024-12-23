#include <ops/declarable/helpers/assign.h>
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename X, typename Z>
static void fastLinearCopy_(const void* vx, void* vz,
                            const sd::LongType length,
                            const sd::LongType start,
                            const sd::LongType stop,
                            sd::LongType xStart,
                            sd::LongType zStart) {
  auto x = reinterpret_cast<const X*>(vx);
  auto z = reinterpret_cast<Z*>(vz);
  for (sd::LongType i = start; i < stop; ++i) {
    z[i + zStart] = static_cast<Z>(x[i + xStart]);
  }
}

template <typename X, typename Z>
static void assign_(const void* vx, const sd::LongType* xShapeInfo,
                    void* vz, const sd::LongType* zShapeInfo,
                    const sd::LongType start,
                    const sd::LongType stop,
                    sd::LongType xStart, sd::LongType zStart) {
  auto x = reinterpret_cast<const X*>(vx);
  auto z = reinterpret_cast<Z*>(vz);

  const sd::LongType xRank = shape::rank(xShapeInfo);
  const sd::LongType zRank = shape::rank(zShapeInfo);
  const sd::LongType* xShape = shape::shapeOf(xShapeInfo);
  const sd::LongType* zShape = shape::shapeOf(zShapeInfo);
  const sd::LongType* xStride = shape::stride(xShapeInfo);
  const sd::LongType* zStride = shape::stride(zShapeInfo);

  sd::LongType xCoords[SD_MAX_RANK];
  sd::LongType zCoords[SD_MAX_RANK];

  for (sd::LongType i = start; i < stop; i++) {
    INDEX2COORDS(i, xRank, xShape, xCoords);
    INDEX2COORDS(i, zRank, zShape, zCoords);

    sd::LongType xOffset, zOffset;
    COORDS2INDEX(xRank, xStride, xCoords, xOffset);
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);

    z[zOffset] = static_cast<Z>(x[xOffset]);
  }
}


SD_LIB_HIDDEN void assign(sd::LaunchContext* context, sd::NDArray* target, sd::NDArray* source) {
  if (target->lengthOf() != source->lengthOf()) {
    std::string errorMsg = "assign helper: Source and target arrays must have the same length. ";
    errorMsg += "Source shape: " + ShapeUtils::shapeAsString(source) + ", ";
    errorMsg += "Target shape: " + ShapeUtils::shapeAsString(target) + ", ";
    errorMsg += "Source datatype: " + DataTypeUtils::asString(source->dataType()) + ", ";
    errorMsg += "Target datatype: " + DataTypeUtils::asString(target->dataType());
    THROW_EXCEPTION(errorMsg.c_str());
  }

  auto xType = source->dataType();
  auto zType = target->dataType();

  const auto length = target->lengthOf();

  bool canUseLinearCopy = !shape::isViewConst(target->shapeInfo()) && !shape::isViewConst(source->shapeInfo())
                          && shape::haveSameShapeAndStrides(source->shapeInfo(), target->shapeInfo());

  if (canUseLinearCopy) {
    auto func = PRAGMA_THREADS_FOR {
      BUILD_DOUBLE_SELECTOR(xType, zType, fastLinearCopy_,
                            (source->dataBuffer()->primary(), target->dataBuffer()->primary(), length, start, stop, source->offset(), target->offset()),
                            SD_COMMON_TYPES, SD_COMMON_TYPES);
    };

    const int numThreads = sd::math::sd_max<int>(1, sd::math::sd_min<int>(length / 1024,
                                                                          sd::Environment::getInstance().maxMasterThreads()));

    samediff::Threads::parallel_for(func, 0, length, 1, numThreads);
  } else {
    auto func = PRAGMA_THREADS_FOR {
      BUILD_DOUBLE_SELECTOR(xType, zType, assign_,
                            (source->dataBuffer()->primary(), source->shapeInfo(), target->dataBuffer()->primary(), target->shapeInfo(), start, stop, source->offset(), target->offset()),
                            SD_COMMON_TYPES, SD_COMMON_TYPES);
    };

    const int numThreads = sd::math::sd_max<int>(1, sd::math::sd_min<int>(length / 1024,
                                                                          sd::Environment::getInstance().maxMasterThreads()));

    samediff::Threads::parallel_for(func, 0, length, 1, numThreads);
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd