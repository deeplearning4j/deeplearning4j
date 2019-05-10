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
//  @author raver119@gmail.com
//

#ifndef LIBND4J_EXTERNALWORKSPACE_H
#define LIBND4J_EXTERNALWORKSPACE_H

#include <pointercast.h>
#include <dll.h>

namespace nd4j {
    namespace memory {
        class ND4J_EXPORT ExternalWorkspace {
        private:
            void *_ptrH = nullptr;
            void *_ptrD = nullptr;

            Nd4jLong _sizeH = 0L;
            Nd4jLong _sizeD = 0L;
        public:
            ExternalWorkspace() = default;
            ~ExternalWorkspace() = default;

            ExternalWorkspace(Nd4jPointer ptrH, Nd4jLong sizeH, Nd4jPointer ptrD, Nd4jLong sizeD);
            
            void *pointerHost();
            void *pointerDevice();

            Nd4jLong sizeHost();
            Nd4jLong sizeDevice();
        };
    }
}

#endif