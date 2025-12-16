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
#include <array/PrimaryPointerDeallocator.h>

#if defined(SD_GCC_FUNCTRACE)
#include <array/ShapeCacheLifecycleTracker.h>
#endif


namespace sd {

ConstantShapeBuffer* CpuShapeBufferCreator::create(const LongType* shapeInfo, int rank) {
    if (shapeInfo == nullptr) {
        THROW_EXCEPTION("CpuShapeBufferCreator::create: shapeInfo cannot be nullptr");
    }
    if (rank < 0 || rank > SD_MAX_RANK) {
        std::string msg = "CpuShapeBufferCreator::create: invalid rank: " + std::to_string(rank);
        THROW_EXCEPTION(msg.c_str());
    }

    const int shapeInfoLength = shape::shapeInfoLength(rank);
    LongType* shapeCopy = new LongType[shapeInfoLength];
    if (shapeCopy == nullptr) {
        THROW_EXCEPTION("CpuShapeBufferCreator::create: failed to allocate memory for shapeCopy");
    }
    std::memcpy(shapeCopy, shapeInfo, shapeInfoLength * sizeof(LongType));

    // Previously used PointerDeallocator (no-op) which leaked memory
    auto deallocator = std::shared_ptr<PrimaryPointerDeallocator>(
        new PrimaryPointerDeallocator(),
        [] (PrimaryPointerDeallocator* ptr) { delete ptr; }
    );

    auto hPtr = new PointerWrapper(shapeCopy, deallocator);
    if (hPtr == nullptr) {
        delete[] shapeCopy;
        THROW_EXCEPTION("CpuShapeBufferCreator::create: failed to create PointerWrapper");
    }

    auto buffer = new ConstantShapeBuffer(hPtr);
    if (buffer == nullptr) {
        delete hPtr;
        THROW_EXCEPTION("CpuShapeBufferCreator::create: failed to create ConstantShapeBuffer");
    }

#if defined(SD_GCC_FUNCTRACE)
    // Track shape cache allocation
    sd::array::ShapeCacheLifecycleTracker::getInstance().recordAllocation(shapeCopy);
#endif

    // Session #977: Validate buffer before returning.
    // primary() will now throw an exception if the buffer is invalid,
    // so we just call it to trigger validation.
#ifdef __cpp_exceptions
    try {
        (void)buffer->primary();  // Trigger validation - will throw if invalid
    } catch (...) {
        delete buffer;
        throw;  // Re-throw the exception
    }
#else
    // Exceptions disabled - direct validation call without try/catch
    (void)buffer->primary();  // Trigger validation
#endif

    return buffer;
}

CpuShapeBufferCreator& CpuShapeBufferCreator::getInstance() {
    static CpuShapeBufferCreator instance;
    return instance;
}

} // namespace sd
