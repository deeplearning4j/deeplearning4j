/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
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
// @author raver119@gmail.com
//

#include "../ConstantShapeHelper.h"

namespace nd4j {
    ConstantShapeHelper::ConstantShapeHelper() {
        //
    }

    ConstantShapeHelper* ConstantShapeHelper::getInstance() {
        return _INSTANCE;
    }

    DataBuffer& ConstantShapeHelper::bufferForShapeInfo(ShapeDescriptor &descriptor) {
        DataBuffer buffer;
        return buffer;
    }

    DataBuffer& ConstantShapeHelper::bufferForShapeInfo(const Nd4jLong *shapeInfo) {
        DataBuffer buffer;
        return buffer;
    }
}