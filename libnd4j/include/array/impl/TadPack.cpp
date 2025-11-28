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
//  @author raver119@gmail.com
//
#include "../TadPack.h"

#include <helpers/shape.h>
#include <system/Environment.h>
#include <sstream>

#if defined(SD_GCC_FUNCTRACE)
#include <array/TADCacheLifecycleTracker.h>
#endif

namespace sd {
TadPack::TadPack( ConstantShapeBuffer *shapes,
                  ConstantOffsetsBuffer *offets, LongType numTads,
                 LongType* dimensions, LongType dimLength)
    : _tadShape(shapes),
      _tadOffsets(offets) {
  _numTads = numTads;
  _dimensionsLength = dimLength;
  if(dimensions != nullptr) {
    _dimensions = new LongType[dimLength];
    for(int i = 0; i < dimLength; i++) {
      _dimensions[i] = dimensions[i];
    }
  }

  computeHash();

#if defined(SD_GCC_FUNCTRACE) && !defined(__JAVACPP_HACK__)
  // Track TAD cache allocation - only capture stack trace if tracking is enabled
  // Stack trace capture is expensive, so skip it when tracking is disabled
  auto& tracker = sd::array::TADCacheLifecycleTracker::getInstance();
  if (tracker.isEnabled()) {
    // Capture stack trace for this TadPack allocation
    _stackTrace = backward::StackTrace();
    _stackTrace.load_here(32);
  }

  // Always record allocation for basic counting (cheap operation)
  size_t shape_info_bytes = 0;
  size_t offsets_bytes = 0;

  if (_tadShape != nullptr && _tadShape->primary() != nullptr) {
    shape_info_bytes = shape::shapeInfoByteLength(_tadShape->primary());
  }

  if (_tadOffsets != nullptr && _tadOffsets->primary() != nullptr) {
    offsets_bytes = _numTads * sizeof(LongType);
  }

  std::vector<LongType> dims;
  if (_dimensions != nullptr) {
    dims.assign(_dimensions, _dimensions + _dimensionsLength);
  }

  tracker.recordAllocation(this, _numTads, shape_info_bytes, offsets_bytes, dims);
#endif

}

TadPack::~TadPack() {
#if defined(SD_GCC_FUNCTRACE) && !defined(__JAVACPP_HACK__)
  // Track TAD cache deallocation before cleanup.
  // Guard must match constructor guard for proper allocation/deallocation pairing.
  sd::array::TADCacheLifecycleTracker::getInstance().recordDeallocation(this);
#endif

  // Clean up dimensions array that was allocated in constructor
  if (_dimensions != nullptr) {
    delete[] _dimensions;
    _dimensions = nullptr;
  }

  // Clean up TAD offsets buffer if we own it
  // This is owned when transferred via releaseOffsets() from TadCalculator
  if (_tadOffsets != nullptr) {
    delete _tadOffsets;
    _tadOffsets = nullptr;
  }

  // DON'T delete _tadShape - it comes from ConstantShapeHelper cache
}

LongType* TadPack::primaryShapeInfo() {
  if(_tadShape->primary() == nullptr)
    THROW_EXCEPTION("TadPack::primaryShapeInfo: primary shape info is nullptr!");
  return _tadShape->primary();
}

LongType* TadPack::primaryOffsets() {
  return _tadOffsets->primary();
}

LongType* TadPack::specialShapeInfo() { return _tadShape->special(); }

LongType* TadPack::specialOffsets() { return _tadOffsets->special(); }

LongType TadPack::numberOfTads() const { return _numTads; }

LongType* TadPack::platformShapeInfo() {
  return Environment::getInstance().isCPU() ? primaryShapeInfo() : specialShapeInfo();
}

LongType* TadPack::platformOffsets() {
  return Environment::getInstance().isCPU() ? primaryOffsets() : specialOffsets();
}


void TadPack::print(const char* msg) {
  printf("---------------------------\n");
  printf("%s: ", msg);
  printf("Offsets:\n");
  for (int e = 0; e < _numTads; e++) {
    printf("%lld, ", _tadOffsets->primary()[e]);
  }
  printf("\n");

  printf("Dimensions:\n");
  if (_dimensions == nullptr || _dimensionsLength == 0) {
    printf("none\n");
  } else {
    for (int i = 0; i < _dimensionsLength; i++) {
      printf("%lld, ", _dimensions[i]);
    }
    printf("\n");
  }

  printf("tad pack shape info:");
  shape::printShapeInfo(_tadShape->primary());
  printf("\n");
  printf("number of tads: %lld\n", _numTads);
  printf("shape info length: %lld\n", _shapeInfoLength);
  printf("---------------------------\n");
}

LongType TadPack::shapeInfoLength() { return shape::shapeInfoLength(primaryShapeInfo()); }
bool TadPack::operator==( TadPack& other)  {
  // Compare number of TADs
  if (_numTads != other._numTads)
    return false;

  // Compare shape information
  LongType* thisShape = primaryShapeInfo();
  LongType* otherShape = other.primaryShapeInfo();

  // Check for null shape info
  if ((thisShape == nullptr) != (otherShape == nullptr))
    return false;

  if (thisShape != nullptr) {
    // Compare rank
    const int thisRank = shape::rank(thisShape);
    const int otherRank = shape::rank(otherShape);
    if (thisRank != otherRank)
      return false;

    // Compare shape order
    if (shape::order(thisShape) != shape::order(otherShape))
      return false;

    // Compare data type
    if (ArrayOptions::dataType(thisShape) != ArrayOptions::dataType(otherShape))
      return false;

    // Compare shape dimensions
    for (int i = 0; i < thisRank; i++) {
      if (shape::shapeOf(thisShape)[i] != shape::shapeOf(otherShape)[i])
        return false;
    }

    // Compare shape strides
    for (int i = 0; i < thisRank; i++) {
      if (shape::stride(thisShape)[i] != shape::stride(otherShape)[i])
        return false;
    }
  }

  // Compare dimensions array
  if ((_dimensions == nullptr) != (other._dimensions == nullptr))
    return false;

  if (_dimensions != nullptr) {
    if (_dimensionsLength != other._dimensionsLength)
      return false;

    for (LongType i = 0; i < _dimensionsLength; i++) {
      if (_dimensions[i] != other._dimensions[i])
        return false;
    }
  }

  // Compare offsets
  LongType* thisOffsets = primaryOffsets();
  LongType* otherOffsets = other.primaryOffsets();

  // Check for null offsets
  if ((thisOffsets == nullptr) != (otherOffsets == nullptr))
    return false;

  if (thisOffsets != nullptr) {
    for (LongType i = 0; i < _numTads; i++) {
      if (thisOffsets[i] != otherOffsets[i])
        return false;
    }
  }

  return true;
}

std::string TadPack::getStackTraceAsString() const {
#if defined(SD_GCC_FUNCTRACE) && !defined(__JAVACPP_HACK__)
  // Use backward::Printer to format the stack trace into a string
  std::ostringstream oss;
  backward::Printer p;
  p.snippet = false;  // Don't include source code snippets
  p.address = true;   // Include addresses
  p.object = false;   // Don't include object file info
  p.color_mode = backward::ColorMode::never;  // No ANSI colors in string

  // Print to our string stream (we need to cast away const to use _stackTrace)
  // This is safe since print doesn't modify the StackTrace
  backward::StackTrace& mutable_st = const_cast<backward::StackTrace&>(_stackTrace);
  p.print(mutable_st, oss);

  return oss.str();
#else
  return "";  // Return empty string when functrace is not enabled
#endif
}


}  // namespace sd
