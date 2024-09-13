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
#if NOT_EXCLUDED(OP_write_list)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
LIST_OP_IMPL(write_list, 2, 1, 0, -2) {
  auto list = INPUT_LIST(0);
  auto output = OUTPUT_VARIABLE(0);
  // nd4j mode
  if (block.width() >= 3) {
    auto input = INPUT_VARIABLE(block.width() - 2);
    auto idx = INPUT_VARIABLE(block.width() - 1);

    REQUIRE_TRUE(idx->isScalar(), 0, "Index should be Scalar");

    Status result = list->write(idx->e<int>(0), new NDArray(input->dup(input->ordering())));

    auto res = NDArrayFactory::create_(list->counter(), block.launchContext());

    setupResult(res, block);
    //                OVERWRITE_RESULT(res);

    return result;
  } else if (block.getIArguments()->size() == 1) {
    auto input = INPUT_VARIABLE(1);
    auto idx = INT_ARG(0);

    Status result = list->write(idx, new NDArray(input->dup(input->ordering())));

    auto res = NDArrayFactory::create_(list->counter(), block.launchContext());
    setupResult(res, block);
    return result;
  } else
    return Status::BAD_INPUT;
}
DECLARE_SYN(TensorArrayWriteV3, write_list);
DECLARE_SYN(tensorarraywritev3, write_list);
}  // namespace ops
}  // namespace sd

#endif
