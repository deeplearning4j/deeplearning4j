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

#include <mutex>
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
    LongType* shapeCopy = new LongType[shapeInfoLength + SD_SHAPE_ALLOC_PADDING]();
    for(int i = 0; i < shapeInfoLength; i++) {
        shapeCopy[i] = shapeInfo[i];
    }

    // Write canary stamps in the padding area to detect buffer overruns.
    // If the data at indices [0..shapeInfoLength-1] gets corrupted but the
    // canaries are intact, the write came from WITHIN the buffer (pointer
    // mutation or reinterpretation). If the canaries are also corrupted,
    // an adjacent allocation overran into this buffer.
    static constexpr LongType SHAPE_CANARY = static_cast<LongType>(0x5AFE5AFE5AFE5AFELL);
    for (int i = 0; i < 8 && (shapeInfoLength + i) < (shapeInfoLength + SD_SHAPE_ALLOC_PADDING); i++) {
        shapeCopy[shapeInfoLength + i] = SHAPE_CANARY;
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
    // Allocate GPU shape info with the same padding as host side. Without this,
    // GPU shape info gets only +8 bytes padding from ALLOCATE_SPECIAL while host
    // gets 32KB. CUDA kernel buffer overruns from neighboring allocations corrupt
    // the tiny GPU shape info → rank reads as garbage → kernel accesses wildly
    // out-of-bounds memory → CUDA error 700 (illegal memory access).
    PointerWrapper* dPtr = new PointerWrapper(
        ConstantHelper::getInstance().replicatePointer(shapeCopy,
                                                   (shapeInfoLength + SD_SHAPE_ALLOC_PADDING) * sizeof(LongType)),
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
    static CudaShapeBufferCreator* instance = nullptr;
    static std::once_flag initFlag;
    std::call_once(initFlag, []() {
        instance = new CudaShapeBufferCreator();
    });
    return *instance;
}

} // namespace sd
