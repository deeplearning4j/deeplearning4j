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

#ifndef SD_PLATFORMHELPER_H
#define SD_PLATFORMHELPER_H

#include <graph/Context.h>
#include <string>
#include <pointercast.h>
#include <dll.h>

namespace  nd4j {
    namespace ops {
        /**
         * This abstract class defines methods used by platform-specific helpers implementations
         */
        class ND4J_EXPORT PlatformHelper {
        protected:
            // name of the operation this helper is built for
            std::string _name;

            // hash of the operation this helper is built for
            Nd4jLong _hash;
        public:
            PlatformHelper(const char *name);
            ~PlatformHelper() = default;

            std::string name();
            Nd4jLong hash();

            /**
             * This method checks, if given helper can be used with given input/output/configuration options
             *
             * @param context
             * @return
             */
            virtual bool isUsable(graph::Context &context) = 0;

            /**
             * This method invokes helper. Typically this method replaces actual op execution
             *
             * @param context
             * @return
             */
            virtual Nd4jStatus invokeHelper(graph::Context &context) = 0;
        };
    }
}


#endif //SD_PLATFORMHELPER_H
