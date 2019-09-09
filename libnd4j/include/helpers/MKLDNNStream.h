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

#ifndef __STANDALONE_BUILD__
#include "config.h"
#endif

#ifdef HAVE_MKLDNN
#include <mkldnn.hpp>

namespace nd4j {
    class MKLDNNStream {
    protected:
        std::string _opName;

        std::vector<const NDArray*> _inputs;
        std::vector<const NDArray*> _outputs;
        std::vector<float> _floatArguments;
        std::vector<int> _intArguments;

        mkldnn::engine _engine = mkldnn::engine(mkldnn::engine::kind::cpu, 0);
        std::vector<mkldnn::primitive> _operations;

    public:
        template <typename X, typename Y>
        static bool isSupported() {
            return typeid(X) == typeid(float) && typeid(Y) == typeid(float);
        }

        static bool isSupported(const std::vector<const NDArray*> &arrays) {
            for (auto i = arrays.begin(); i != arrays.end(); i++) {
                if (*i != nullptr && (*i)->dataType() != nd4j::DataType::FLOAT32) {
                    return false;
                }
            }
            return true;
        }

        MKLDNNStream(const std::string &opName) : _opName(opName) { }

        bool checkAndReset(const std::vector<const NDArray*> &inputs, const std::vector<const NDArray*> &outputs,
                const std::vector<float> &floatArguments, const std::vector<int> &intArguments) {
            if (inputs != _inputs || outputs != _outputs || floatArguments != _floatArguments || intArguments != _intArguments) {
                _inputs = inputs;
                _outputs = outputs;
                _floatArguments = floatArguments;
                _intArguments = intArguments;
                _operations.clear();
                return true;
            }
            return false;
        }

        const mkldnn::engine &getEngine() { return _engine; }
        void setEngine(const mkldnn::engine &engine) { _engine = engine; }
    };
}
#endif

#endif //LIBND4J_MKLDNNSTREAM_H
