/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// Created by saudet on 8/30/2018.
//

#ifndef LIBND4J_MKLDNNSTREAM_H
#define LIBND4J_MKLDNNSTREAM_H

#if !defined(__STANDALONE_BUILD__)
#include "config.h"
#endif

#if defined(HAVE_MKLDNN)

namespace nd4j {
    class MKLDNNStream {
    protected:
        std::string _opName;

        std::vector<const NDArray*> _inputs;
        std::vector<const NDArray*> _outputs;
        std::vector<float> _floatArguments;
        std::vector<int> _intArguments;

    public:
        template <typename X, typename Y>
        static bool isSupported() {
            // FIXME: strict float support doesn't work anymore
            return typeid(X) == typeid(float) && typeid(Y) == typeid(float);
        }

        static bool isSupported(const std::vector<const NDArray*> &arrays) {
            // FIXME: strict float support doesn't work anymore
            for (auto v:arrays) {
                if (v != nullptr && v->dataType() != nd4j::DataType::FLOAT32) {
                    return false;
                }
            }
            return true;
        }

        explicit MKLDNNStream(const std::string &opName) : _opName(opName) { }

        bool checkAndReset(const std::vector<const NDArray*> &inputs, const std::vector<const NDArray*> &outputs,
                const std::vector<float> &floatArguments, const std::vector<int> &intArguments) {
            if (inputs != _inputs || outputs != _outputs || floatArguments != _floatArguments || intArguments != _intArguments) {
                _inputs = inputs;
                _outputs = outputs;
                _floatArguments = floatArguments;
                _intArguments = intArguments;
                return true;
            }
            return false;
        }
    };
}
#endif

#endif //LIBND4J_MKLDNNSTREAM_H
