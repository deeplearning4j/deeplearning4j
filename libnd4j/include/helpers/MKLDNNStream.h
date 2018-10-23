/*******************************************************************************
 * Copyright (c) 2018 Skymind, Inc.
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

#ifdef HAVE_MKLDNN
#include <mkldnn.hpp>

namespace nd4j {
    template <typename T> class MKLDNNStream {
    protected:
        std::string _opName;

        std::vector<NDArray<T>*> _inputs;
        std::vector<NDArray<T>*> _outputs;
        std::vector<T> _floatArguments;
        std::vector<int> _intArguments;

        mkldnn::engine _engine = mkldnn::engine(mkldnn::engine::cpu, 0);
        std::vector<mkldnn::memory> _memory;
        mkldnn::primitive _operation;

    public:
        static bool isSupported() { return typeid(T) == typeid(float); }

        MKLDNNStream(const std::string &opName) : _opName(opName) { }

        bool checkAndReset(const std::vector<NDArray<T>*> &inputs, const std::vector<NDArray<T>*> &outputs,
                const std::vector<T> &floatArguments, const std::vector<int> &intArguments) {
            if (inputs != _inputs || outputs != _outputs || floatArguments != _floatArguments || intArguments != _intArguments) {
                _inputs = inputs;
                _outputs = outputs;
                _floatArguments = floatArguments;
                _intArguments = intArguments;
                _operation.reset(nullptr);
                _memory.clear();
                return true;
            }
            return false;
        }

        const mkldnn::engine &getEngine() { return _engine; }
        void setEngine(const mkldnn::engine &engine) { _engine = engine; }

        const std::vector<mkldnn::memory> &getMemory() { return _memory; }
        void setMemory(const std::vector<mkldnn::memory> &memory) { _memory = memory; }

        const mkldnn::primitive &getOperation() { return _operation; }
        void setOperation(const mkldnn::primitive &operation) { _operation = operation; }

        bool submitAndWait(mkldnn::stream::kind kind = mkldnn::stream::kind::eager) {
            nd4j_debug("Executing %s with MKL-DNN\n", _opName.c_str());
            // need to create a new one because already executed streams become unusable
            mkldnn::stream stream(kind);
            return stream.submit({_operation}).wait();
        }
    };
}
#endif

#endif //LIBND4J_MKLDNNSTREAM_H
