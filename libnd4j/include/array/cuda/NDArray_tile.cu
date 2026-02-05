/* ******************************************************************************
 * NDArray_tile.cu - Tile operations
 * Split from NDArray.cu to reduce object file size for large binary builds
 ******************************************************************************/

#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <array/DataBuffer.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <loops/special_kernels.h>

namespace sd {

//////////////////////////////////////////////////////////////////////////
NDArray NDArray::tile(const std::vector<LongType>& reps) {
  int dim = reps.size();
  LongType product = 1;
  for (const auto& item : reps) product *= item;

  if (product < 1) THROW_EXCEPTION("NDArray::tile method: one of the elements in reps array is zero !");

  int rankOld = rankOf();
  int diff = rankOld - dim;

  if (product == 1) {
    NDArray result(*this);
    if (diff < 0) {
      std::vector<LongType> shapeNew = reps;
      memcpy(&shapeNew[-diff], result.shapeInfo() + 1, rankOld * sizeof(LongType));
      result.reshapei(ordering(), shapeNew);
    }
    return result;
  }

  auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, getContext()->getWorkspace());
  DataBuffer* newBuff = new DataBuffer(shape::length(newShapeInfo) * sizeOfT(),
                                        dataType(), getContext()->getWorkspace(), true);
  NDArray result(newBuff, const_cast<sd::LongType*>(newShapeInfo), getContext());

  const auto resultLen = result.lengthOf();
  auto xType = this->dataType();
  auto stream = getContext()->getCudaStream();

  prepareSpecialUse({&result}, {this});
  BUILD_SINGLE_SELECTOR(xType, tileKernelH,
                        (this->specialBuffer(), this->specialShapeInfo(), result.specialBuffer(),
                            result.specialShapeInfo(), resultLen, stream),
                        SD_COMMON_TYPES);
  registerSpecialUse({&result}, {this});

  return result;
}

//////////////////////////////////////////////////////////////////////////
void NDArray::tile(const std::vector<LongType>& reps, NDArray& target) {
  auto repProd = shape::prodLong(reps.data(), reps.size());
  if (repProd < 1) THROW_EXCEPTION("NDArray::tile: reps can't contain 0s");

  auto newShapeInfo = ShapeUtils::evalTileShapeInfo(*this, reps, getContext()->getWorkspace());
  if (!shape::equalsSoft(newShapeInfo, target.shapeInfo())) {
    THROW_EXCEPTION("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");
  }

  const int targetLen = target.lengthOf();
  auto stream = getContext()->getCudaStream();

  prepareSpecialUse({&target}, {this});
  BUILD_SINGLE_SELECTOR_TWICE(
      target.dataType(), tileKernelHH,
      (specialBuffer(), specialShapeInfo(), target.specialBuffer(), target.specialShapeInfo(), targetLen, stream),
      SD_COMMON_TYPES);
  registerSpecialUse({&target}, {this});
}

//////////////////////////////////////////////////////////////////////////
void NDArray::tile(NDArray& target) {
  if (rankOf() > target.rankOf())
    THROW_EXCEPTION(
        "NDArray::tile method - rank of target array must be bigger or equal to the rank of this array !");

  if (!ShapeUtils::areShapesBroadcastable(*this, target))
    THROW_EXCEPTION("NDArray::tile method - shapeInfo of target array is not suitable for tile operation !");

  const auto targetLen = target.lengthOf();
  auto stream = getContext()->getCudaStream();

  prepareSpecialUse({&target}, {this});
  BUILD_SINGLE_SELECTOR_TWICE(
      target.dataType(), tileKernelHH,
      (specialBuffer(), specialShapeInfo(), target.specialBuffer(), target.specialShapeInfo(), targetLen, stream),
      SD_COMMON_TYPES);
  registerSpecialUse({&target}, {this});
}

}  // namespace sd
