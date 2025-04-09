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

namespace sd {

ConstantShapeBuffer* CudaShapeBufferCreator::create(const LongType* shapeInfo, int rank) {
    const int shapeInfoLength = shape::shapeInfoLength(rank);
    LongType* shapeCopy = new LongType[shapeInfoLength];
    for(int i = 0; i < shapeInfoLength; i++) {
        shapeCopy[i] = shapeInfo[i];
    }

    auto deallocator = std::shared_ptr<CudaPointerDeallocator>(
        new CudaPointerDeallocator(),
        [] (CudaPointerDeallocator* ptr) {
          delete ptr;
        }
    );
    
    auto hPtr = new PointerWrapper(shapeCopy, deallocator);
    
    // Create device pointer for CUDA
    auto dPtr =new PointerWrapper(
        ConstantHelper::getInstance().replicatePointer(shapeCopy,
                                                   shapeInfoLength * sizeof(LongType)),
        std::make_shared<CudaPointerDeallocator>());

    if(dPtr->pointer() == nullptr) {
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
