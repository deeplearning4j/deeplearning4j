/* ******************************************************************************
*
* Copyright (c) 2024 Konduit K.K.
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
// @author Adam Gibson
//

#ifndef DEV_TESTS_TADCALCULATOR_H
#define DEV_TESTS_TADCALCULATOR_H

#include <array/TadPack.h>
#include <system/common.h>
#include <helpers/ConstantHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <array/ConstantShapeBuffer.h>
#include <array/ConstantOffsetsBuffer.h>
#include <vector>
#include <memory>

namespace sd {

/**
* TadCalculator handles the computation of Tensor Along Dimension (TAD) information
* including shapes and offsets for sub-arrays.
 */
class SD_LIB_EXPORT TadCalculator {
 private:
  LongType* _originalShape;       // Original shape info pointer
  ConstantShapeBuffer *_tadShape;        // Calculated TAD shape buffer
  ConstantOffsetsBuffer *_tadOffsets;    // Calculated TAD offsets buffer
  LongType _numTads;                    // Number of TADs

 public:
  /**
    * Constructor for TadCalculator
    * @param originalShape Pointer to the original shape information
   */
  explicit TadCalculator(LongType* originalShape);
  ~TadCalculator();

  /**
    * Creates a TAD pack for the given dimensions
    * @param dimensions Vector of dimensions to calculate TADs for
   */
  void createTadPack(const std::vector<LongType>& dimensions);

  /**
    * Returns the calculated TAD shape buffer
    * @return ConstantShapeBuffer containing TAD shape information
   */
  ConstantShapeBuffer *tadShape() const { return _tadShape; }

  /**
    * Returns the calculated TAD offsets buffer
    * @return ConstantOffsetsBuffer containing TAD offset information
   */
  ConstantOffsetsBuffer *tadOffsets() const { return _tadOffsets; }

  /**
    * Releases ownership of the TAD offsets buffer to the caller.
    * 
    * Usage Pattern:
    * This method should be used when transferring ownership of the offsets buffer to another
    * object that will manage its lifetime (e.g., TadPack). This is typically done immediately
    * after calling createTadPack() to construct a TadPack with the calculated offsets.
    * 
    * Lifecycle Implications:
    * - After this call, the TadCalculator no longer owns the offsets and will not delete them
    * - The caller becomes responsible for managing and deleting the returned buffer
    * - The internal _tadOffsets pointer is set to nullptr
    * 
    * IMPORTANT: After calling releaseOffsets(), the TadCalculator object should not be used
    * for any operations that depend on the offsets buffer. Calling tadOffsets() will return
    * nullptr, and the object should generally be destroyed or reset soon after releasing.
    * 
    * Example Usage:
    * @code
    *   // Assume we have a shape info pointer and dimensions vector
    *   LongType* shapeInfo = ...;
    *   std::vector<LongType> dimensions = {0, 1};
    *   
    *   TadCalculator calculator(shapeInfo);
    *   calculator.createTadPack(dimensions);
    *   auto pack = std::make_shared<TadPack>(
    *       calculator.tadShape(),
    *       calculator.releaseOffsets(),  // Transfer ownership to TadPack
    *       calculator.numberOfTads(),
    *       dimensions.data(),
    *       dimensions.size());
    *   // Calculator should not be used after this point
    * @endcode
    * 
    * @return ConstantOffsetsBuffer pointer that the caller must manage, or nullptr if
    *         offsets have not been calculated or have already been released
   */
  ConstantOffsetsBuffer* releaseOffsets() {
    ConstantOffsetsBuffer* temp = _tadOffsets;
    _tadOffsets = nullptr;  // Release ownership
    return temp;
  }

  /**
    * Returns the number of TADs calculated
    * @return Number of TADs
   */
  LongType numberOfTads() const { return _numTads; }
};

} // namespace sd

#endif // DEV_TESTS_TADCALCULATOR_H
