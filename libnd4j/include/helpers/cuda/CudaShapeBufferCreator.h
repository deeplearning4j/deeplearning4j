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

#ifndef LIBND4J_CUDASHAPEBUFFERCREATOR_H
#define LIBND4J_CUDASHAPEBUFFERCREATOR_H

#include <helpers/ShapeBufferCreator.h>
#include <helpers/shape.h>
#include <memory>

namespace sd {

/**
 * CUDA implementation of the ShapeBufferCreator.
 * Creates shape buffers with both host and device pointers.
 */
class CudaShapeBufferCreator : public ShapeBufferCreator {
public:
    /**
     * Create a ConstantShapeBuffer for CUDA usage
     */
    ConstantShapeBuffer* create(const LongType* shapeInfo, int rank) override;
    
    // Singleton pattern implementation
    static CudaShapeBufferCreator& getInstance();

private:
    CudaShapeBufferCreator() = default;
};

} // namespace sd

#endif // LIBND4J_CUDASHAPEBUFFERCREATOR_H
