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

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_unstack_list)

#include <ops/declarable/headers/list.h>

namespace nd4j {
namespace ops {
    LIST_OP_IMPL(unstack_list, 1, 1, 0, 0) {
        auto input = INPUT_VARIABLE(0);

        auto list = new NDArrayList(0, true);
        list->unstack(input, 0);

        OVERWRITE_RESULT(list);

        return Status::OK();
    }
}
}

#endif