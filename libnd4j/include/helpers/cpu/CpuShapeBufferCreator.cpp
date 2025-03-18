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

#include <helpers/cpu/CpuShapeBufferCreator.h>


namespace sd {

ConstantShapeBuffer* CpuShapeBufferCreator::create(const LongType* shapeInfo, int rank) {
    const int shapeInfoLength = shape::shapeInfoLength(rank);
    LongType* shapeCopy = new LongType[shapeInfoLength];
    std::memcpy(shapeCopy, shapeInfo, shapeInfoLength * sizeof(LongType));
    
    auto deallocator = std::shared_ptr<PointerDeallocator>(
        new PointerDeallocator(),
        [] (PointerDeallocator* ptr) { delete ptr; }
    );
    
    auto hPtr = std::make_shared<PointerWrapper>(shapeCopy, deallocator);
    auto buffer = new ConstantShapeBuffer(hPtr);
    
    return buffer;
}

CpuShapeBufferCreator& CpuShapeBufferCreator::getInstance() {
    static CpuShapeBufferCreator instance;
    return instance;
}

} // namespace sd
