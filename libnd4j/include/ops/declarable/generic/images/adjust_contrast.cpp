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
//  @author George A. Shulinok <sgazeos@gmail.com>
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_adjust_contrast)

#include <array/NDArrayFactory.h>
#include <ops/declarable/headers/parity_ops.h>

namespace sd {
namespace ops {

////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(adjust_contrast, 1, 1, true, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // just skip op if input is empty
  if (input->isEmpty()) return sd::Status::OK;

  REQUIRE_TRUE(block.numT() > 0 || block.width() > 1, 0, "ADJUST_CONTRAST: Scale factor required");
  REQUIRE_TRUE(input->rankOf() > 2, 0, "ADJUST_CONTRAST: op expects rank of input array to be >= 3, but got %i instead",
               input->rankOf());


  NDArray* factor = nullptr;

  if (block.width() > 1)
    factor = INPUT_VARIABLE(1);
  else {
    factor = new NDArray(output->dataType(), block.launchContext());
    factor->p(0, T_ARG(0));
  }

  // fill up axes vector first
  std::vector<LongType> axes(input->rankOf() - 1);
  for (auto i = 0; i < axes.size(); ++i) axes[i] = i;
  // mean as reduction for last dimension set
  auto mean = input->reduceAlongDimension(reduce::Mean, &axes);
  auto part1 = (*input - mean);
  auto part2 = part1 * *factor;
  auto part3 = part2 + mean;
  // this is contrast calculation
  output->assign(part3);


  return sd::Status::OK;
}

DECLARE_TYPES(adjust_contrast) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS})->setSameMode(true);
}

////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(adjust_contrast_v2, 1, 1, true, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // just skip op if input is empty
  if (input->isEmpty()) return sd::Status::OK;

  REQUIRE_TRUE(input->rankOf() > 2, 0,
               "ADJUST_CONTRAST_V2: op expects rank of input array to be >= 3, but got %i instead", input->rankOf());
  REQUIRE_TRUE(block.numT() > 0 || block.width() > 1, 0, "ADJUST_CONTRAST_V2: Scale factor required");

  NDArray* factor = nullptr;
  auto size = input->sizeAt(-2) * input->sizeAt(-3);
  auto channels = input->sizeAt(-1);
  auto batch = input->lengthOf() / (size * channels);
  auto input3D = input->reshape(input->ordering(), {batch, size, channels});
  auto output3D = input->reshape(input->ordering(), {batch, size, channels});

  if (block.width() > 1) {
    sd_print("First factor\n");
    //TODO: figure out why this value is sometimes corrupted
    //despite loading correctly
    //we know that this array is correct right up to execution
    //1 suspect is context closing?
    //I do sometimes see odd things like ops being executed twice.
    //there could be some sort of reuse going on that I'm not seeing yet.
    factor = INPUT_VARIABLE(1);
    factor->syncToDevice();
    factor->syncToHost();
  }
  else {
    sd_print("Factor -> p\n");
    factor = new NDArray(output->dataType(), block.launchContext());
    factor->p(0, T_ARG(0));
  }

  std::vector<LongType> axes({1});  // dim 1 of pseudoresult

  sd_print("Before mean\n");
  // mean as reduction for last dimension set over size (dim 1) of result3D
  auto mean = input3D.reduceAlongDimension(reduce::Mean, &axes);
  mean.printIndexedBuffer("Mean buffer\n");
  sd_print("After mean\n");
  // result as (x - mean) * factor + mean
  auto temp = input3D.ulike();
  temp.printIndexedBuffer("Temp created\n");
  sd_print("Created temp\n");
  std::vector<sd::LongType> zeroTwo = {0, 2};
  input3D.printIndexedBuffer("Input 3d before apply");
  input3D.applyBroadcast(broadcast::Subtract,&zeroTwo, mean, temp);
  input3D.printIndexedBuffer("Input3d after subtract\n");
  sd_print("Applied subtract\n");
  temp.printIndexedBuffer("Temp buffer before multiply");
  factor->printIndexedBuffer("Factor before multiply");
  temp.applyScalarArr(scalar::Multiply, *factor, temp);
  factor->printIndexedBuffer("Factor after multiply");
  temp.printIndexedBuffer("Temp buffer after multiply\n");
  sd_print("Applied multiply\n");
  temp.applyBroadcast(broadcast::Add, &zeroTwo, mean, output3D);
  temp.printIndexedBuffer("Temp buffer after zadd\n");
  output3D.printIndexedBuffer("OUTPUT 3d indexed buffer\n");
  sd_print("Applied add\n");
  output3D.printCurrentBuffer<float>(false,"Output3D current buffer device \n");
  output3D.printCurrentBuffer<float>(true,"Output3D current buffer host\n");

  output->assign(output3D);
  output->synchronize("");
  output->printCurrentBuffer<float>(false,"Output current buffer device \n");
  output->printCurrentBuffer<float>(true,"Output current buffer host\n");

  sd_print("Assigned output\n");

  return sd::Status::OK;
}

DECLARE_TYPES(adjust_contrast_v2) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS})->setSameMode(true);
}

}  // namespace ops
}  // namespace sd

#endif
