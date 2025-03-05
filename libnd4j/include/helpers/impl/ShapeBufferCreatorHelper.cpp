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

#include <helpers/ShapeBufferCreatorHelper.h>
#include <helpers/cpu/CpuShapeBufferCreator.h>
#include <stdexcept>

namespace sd {

// Initialize static member
ShapeBufferCreator* ShapeBufferCreatorHelper::currentCreator_ = nullptr;

ShapeBufferCreator& ShapeBufferCreatorHelper::getCurrentCreator() {
    return *currentCreator_;
}

void ShapeBufferCreatorHelper::setCurrentCreator(ShapeBufferCreator* creator) {
    if (creator == nullptr) {
        throw std::invalid_argument("ShapeBufferCreator cannot be null");
    }
    currentCreator_ = creator;
}

} // namespace sd
