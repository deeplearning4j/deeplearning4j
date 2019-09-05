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

#include "../PlatformHelper.h"

namespace nd4j {
    namespace ops {
        PlatformHelper::PlatformHelper(const char *name) {
            // we just store name/hash of target operation
            _name = std::string(name);
            _hash = HashHelper::getInstance()->getLongHash(_name);
        }

        std::string PlatformHelper::name() {
            return _name;
        }

        Nd4jLong PlatformHelper::hash() {
            return _hash;
        }
    }
}