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
// @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_unstack_list)

#include <ops/declarable/headers/list.h>

namespace sd {
namespace ops {
    LIST_OP_IMPL(unstack_list, 1, 1, 0, 0) {
        auto outputList = INPUT_LIST(0);
        auto input = INPUT_VARIABLE(int(outputList != nullptr) );

        if (outputList == nullptr) {
            outputList = new NDArrayList(0, true);
            //block.trackList(outputList);
            setupResultList(outputList, block);
        }
        outputList->unstack(input, INT_ARG(0));

        //OVERWRITE_RESULT(list);

        //
        return Status::OK();
    }
}
}

#endif