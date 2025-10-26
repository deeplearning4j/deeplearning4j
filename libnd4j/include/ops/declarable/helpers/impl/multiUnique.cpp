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
//  @author sgazeos@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_unique)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/multiUnique.h>

namespace sd {
namespace ops {
namespace helpers {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool multiUnique(std::vector<NDArray*> const& inputList, sd::memory::Workspace* workspace) {
  sd::LongType length = 0;
  std::vector<NDArray*> reshaped(inputList.size());
  int pos = 0;
  sd::LongType axis = 0;
  Context cContext(1);
  for (auto array : inputList) {
    if (array->dataType() != sd::DataType::INT32)
      THROW_EXCEPTION("multiUnique: this op support INT32 data type only.");

    std::vector<sd::LongType> reshape = {-1};
    reshaped[pos] = array->reshape(array->ordering(), reshape);
    cContext.setInputArray(pos, reshaped[pos]);

    length += array->lengthOf();
    pos++;
  }
  std::vector<LongType> shape = {length};
  NDArray arrayFull('c',shape, sd::DataType::INT32, inputList[0]->getContext());
  cContext.setOutputArray(0, &arrayFull);
  cContext.setIArguments(&axis, 1);

  sd::ops::concat opConcat;
  auto cResult = opConcat.execute(&cContext);
  if (sd::Status::OK != cResult) THROW_EXCEPTION("multiUnique: cannot execute concat op properly.");

  sd::ops::unique opUnique;
  auto uResult = opUnique.evaluate({&arrayFull});
  if (sd::Status::OK != uResult.status()) THROW_EXCEPTION("multiUnique: cannot execute unique op properly.");

  auto uniqueVals = uResult.at(0);

  bool res = uniqueVals->lengthOf() == arrayFull.lengthOf();

  return res;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
