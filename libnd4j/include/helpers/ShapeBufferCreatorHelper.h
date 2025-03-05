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

#ifndef LIBND4J_SHAPEBUFFERCREATORHELPER_H
#define LIBND4J_SHAPEBUFFERCREATORHELPER_H

#include <helpers/ShapeBufferCreator.h>
#include <exception>

namespace sd {

/**
 * Helper class to manage ShapeBufferCreator instances and provide global access
 */
class ShapeBufferCreatorHelper {
public:
    /**
     * Get the current ShapeBufferCreator instance
     */
    static ShapeBufferCreator& getCurrentCreator();
    
    /**
     * Set the current ShapeBufferCreator to use
     */
    static void setCurrentCreator(ShapeBufferCreator* creator);
    
private:
    static ShapeBufferCreator* currentCreator_;
};

} // namespace sd

#endif // LIBND4J_SHAPEBUFFERCREATORHELPER_H
