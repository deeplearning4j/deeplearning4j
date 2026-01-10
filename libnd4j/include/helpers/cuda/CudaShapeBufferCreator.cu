/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <helpers/ConstantHelper.h>
#include <helpers/cuda/CudaShapeBufferCreator.h>

#include "array/CudaPointerDeallocator.h"
#include "array/PrimaryPointerDeallocator.h"

#include <string>

namespace sd {

ConstantShapeBuffer* CudaShapeBufferCreator::create(const LongType* shapeInfo, int rank) {
    // Validate input shapeInfo before copying
    if (shapeInfo == nullptr) {
        THROW_EXCEPTION("CudaShapeBufferCreator::create: shapeInfo is null");
    }
    if (rank < 0 || rank > SD_MAX_RANK) {
        std::string errorMessage = "CudaShapeBufferCreator::create: invalid rank: ";
        errorMessage += std::to_string(rank);
        THROW_EXCEPTION(errorMessage.c_str());
    }

    // Validate the input shapeInfo data
    LongType inputRank = shapeInfo[0];
    if (inputRank != rank) {
        std::string errorMessage = "CudaShapeBufferCreator::create: shapeInfo rank mismatch. Expected: ";
        errorMessage += std::to_string(rank);
        errorMessage += ", found in shapeInfo[0]: ";
        errorMessage += std::to_string(inputRank);
        THROW_EXCEPTION(errorMessage.c_str());
    }

    const int shapeInfoLength = shape::shapeInfoLength(rank);
    LongType* shapeCopy = new LongType[shapeInfoLength];
    for(int i = 0; i < shapeInfoLength; i++) {
        shapeCopy[i] = shapeInfo[i];
    }

    // Verify copy is correct (the rank at index 0 should match)
    if (shapeCopy[0] != rank) {
        delete[] shapeCopy;
        std::string errorMessage = "CudaShapeBufferCreator::create: copy verification failed. Expected rank: ";
        errorMessage += std::to_string(rank);
        errorMessage += ", copied value: ";
        errorMessage += std::to_string(shapeCopy[0]);
        THROW_EXCEPTION(errorMessage.c_str());
    }

    // Host pointer uses PrimaryPointerDeallocator (delete[])
    auto hostDeallocator = std::shared_ptr<PrimaryPointerDeallocator>(
        new PrimaryPointerDeallocator(),
        [] (PrimaryPointerDeallocator* ptr) {
          delete ptr;
        }
    );

    PointerWrapper* hPtr = new PointerWrapper(shapeCopy, hostDeallocator);

    // Create device pointer for CUDA
    PointerWrapper* dPtr = new PointerWrapper(
        ConstantHelper::getInstance().replicatePointer(shapeCopy,
                                                   shapeInfoLength * sizeof(LongType)),
        std::make_shared<CudaPointerDeallocator>());

    if(dPtr->pointer() == nullptr) {
        delete hPtr;
        delete dPtr;
        THROW_EXCEPTION("Failed to allocate device memory for shape buffer");
    }
    ConstantShapeBuffer *buffer = new ConstantShapeBuffer(hPtr, dPtr);

    return buffer;
}

CudaShapeBufferCreator& CudaShapeBufferCreator::getInstance() {
    static CudaShapeBufferCreator instance;
    return instance;
}

} // namespace sd
