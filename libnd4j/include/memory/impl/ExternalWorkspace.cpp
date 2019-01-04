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

#include <memory/ExternalWorkspace.h>

namespace nd4j {
    namespace memory {
        ExternalWorkspace::ExternalWorkspace(Nd4jPointer ptrH, Nd4jLong sizeH, Nd4jPointer ptrD, Nd4jLong sizeD) {
            _ptrH = ptrH;
            _sizeH = sizeH;

            _ptrD = ptrD;
            _sizeD = sizeD;
        };

        void* ExternalWorkspace::pointerHost() {
            return _ptrH;
        }

        void* ExternalWorkspace::pointerDevice() {
            return _ptrD;
        }

        Nd4jLong ExternalWorkspace::sizeHost() {
            return _sizeH;
        }
        
        Nd4jLong ExternalWorkspace::sizeDevice() {
            return _sizeD;
        }
    }
}