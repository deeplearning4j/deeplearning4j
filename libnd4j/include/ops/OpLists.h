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

#ifndef DEV_TESTS_OPLISTS_H
#define DEV_TESTS_OPLISTS_H

#include <op_boilerplate.h>
#include <vector>
#include <string>
#include <pointercast.h>
#include <legacy_ops.h>

namespace nd4j {
    class LegacyOp {
    protected:
    public:
        int _opNum = -1;
        std::string _opName;

        LegacyOp() = default;
        ~LegacyOp() = default;

        LegacyOp(int opNum, const char *opName) {
            _opNum = opNum;
            _opName = opName;
        }
    };

    class OpLists {
    public:
        static FORCEINLINE std::vector<LegacyOp> transformStrict() {
            std::vector<LegacyOp> ops;

            BUILD_OPLIST(ops, TRANSFORM_STRICT_OPS);

            return op;
        }
    };
}

#endif //DEV_TESTS_OPLISTS_H
