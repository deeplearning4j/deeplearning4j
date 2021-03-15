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
// @author raver119@gmail.com, created on 10.02.18.
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_tf_atan2)

#include <ops/declarable/headers/broadcastable.h>

namespace sd {
namespace ops {

BROADCASTABLE_OP_IMPL(tf_atan2, 0, 0) {

    auto y = INPUT_VARIABLE(0);
    auto x = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    BROADCAST_CHECK_EMPTY(x,y,z);

    // auto tZ = BroadcastHelper<T>::template broadcastApply<simdOps::Atan2<T>>(y, x, z);
    x->applyTrueBroadcast(sd::BroadcastOpsTuple::custom(scalar::Atan2, pairwise::Atan2, broadcast::Atan2), *y, *z, true);

    // if (tZ == nullptr)
    //     return ND4J_STATUS_KERNEL_FAILURE;
    // else if (tZ != z) {
    //     OVERWRITE_RESULT(tZ);
    // }

    return Status::OK();
}

    DECLARE_TYPES(tf_atan2) {
        getOpDescriptor()
                ->setAllowedInputTypes(0, DataType::ANY)
                ->setAllowedInputTypes(1, DataType::ANY)
                ->setAllowedOutputTypes(0, DataType::INHERIT);
    }

}
}

#endif