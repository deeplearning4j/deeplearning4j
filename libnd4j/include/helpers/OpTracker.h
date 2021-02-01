/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#ifndef LIBND4J_OP_TRACKER_H
#define LIBND4J_OP_TRACKER_H

#include <map>
#include <vector>
#include <atomic>
#include <system/pointercast.h>
#include <graph/generated/utils_generated.h>
#include <ops/declarable/OpDescriptor.h>
#include <system/dll.h>

namespace sd {
    class ND4J_EXPORT OpTracker {
    private:
        std::string _export;

        int _operations = 0;
        std::map<sd::graph::OpType, std::vector<sd::ops::OpDescriptor>> _map;

        OpTracker() = default;
        ~OpTracker() = default;

        template <typename T>
        std::string local_to_string(T value);
    public:
        static OpTracker& getInstance();

        int totalGroups();
        int totalOperations();

        void storeOperation(sd::graph::OpType opType, const sd::ops::OpDescriptor& descriptor);
        void storeOperation(sd::graph::OpType opType, const char* opName, const Nd4jLong opNum);

        const char* exportOperations();
    };
}

#endif
