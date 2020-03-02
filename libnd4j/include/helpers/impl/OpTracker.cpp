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

#include <helpers/OpTracker.h>
#include <sstream>
#include <helpers/logger.h>
#include <legacy/NativeOps.h>


using namespace sd::ops;
using namespace sd::graph;

namespace sd {
    
    OpTracker* OpTracker::getInstance() {
        if (_INSTANCE == 0)
            _INSTANCE = new OpTracker();

        return _INSTANCE;
    }

    void OpTracker::storeOperation(sd::graph::OpType opType, const OpDescriptor& descriptor) {
        // check out CPU features
        if (!::isMinimalRequirementsMet()) {

            auto binaryLevel = ::binaryLevel();
            auto optimalLevel = ::optimalLevel();

            switch (binaryLevel) {
                case 3: {
                        nd4j_printf("libnd4j binary was built with AVX512 support, but current CPU doesn't have this instruction set. Exiting now...","");
                    }
                    break;
                case 2: {
                        nd4j_printf("libnd4j binary was built with AVX/AVX2 support, but current CPU doesn't have this instruction set. Exiting now...","");
                    }
                    break;
                default:  {
                    nd4j_printf("Unknown binary validation error. Exiting now...","");
                    }
                    break;
            }

            // we're exiting now
            exit(119);
        }
        //
        if (_map.count(opType) < 1) {
            std::vector<OpDescriptor> vec;
            _map[opType] = vec;
        }

        _operations++;

        auto vec = _map[opType];

        if (std::find(vec.begin(), vec.end(), descriptor) == vec.end())
            _map[opType].emplace_back(descriptor);
    }

    void OpTracker::storeOperation(sd::graph::OpType opType, const char* opName, const Nd4jLong opNum) {
        OpDescriptor descriptor(0, opName, false);
        descriptor.setOpNum((int) opNum);
        descriptor.setHash(-1);

        storeOperation(opType, descriptor);
    }


    template <typename T>
    std::string OpTracker::local_to_string(T value) {
        std::ostringstream os ;
        os << value ;
        return os.str() ;
    }


    int OpTracker::totalGroups() {
        return (int) _map.size();
    }

    int OpTracker::totalOperations() {
        return _operations;
    }

    const char* OpTracker::exportOperations() {
        if (_export.length() == 0) {
            for (auto &v: _map) {
                std::string block = local_to_string(v.first) + " ";

                for (auto &i: v.second) {
                    block += local_to_string(i.getHash()) + ":";
                    block += local_to_string(i.getOpNum()) + ":";
                    block += *i.getOpName() + "<<";
                }

                block += ">>";
                _export += block;
            }
        }

        return _export.c_str();
    }

    sd::OpTracker* sd::OpTracker::_INSTANCE = 0;
}
