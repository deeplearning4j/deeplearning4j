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

#ifndef LIBND4J_SHAPEBUFFERPLATFORMHELPER_H
#define LIBND4J_SHAPEBUFFERPLATFORMHELPER_H

#include <helpers/ShapeBufferCreatorHelper.h>

namespace sd {

/**
 * Platform-specific initialization helper
 * Takes care of setting up the correct creators based on the available hardware
 */
class ShapeBufferPlatformHelper {
public:
    /**
     * Initialize platform-specific components
     * This method should be called during the application startup
     * to ensure proper creators are set based on the available hardware
     */
    static void initialize();

    /**
     * Automatic initialization through static initialization
     * C++17 guarantees thread-safety for static initialization
     */
    static inline const bool initialized = (initialize(), true);

private:
 ShapeBufferPlatformHelper() = delete;  // Prevent instantiation
};

} // namespace sd

#endif  // LIBND4J_SHAPEBUFFERPLATFORMHELPER_H
