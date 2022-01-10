/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>

#include "vednnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(maxpool2d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  auto kH = INT_ARG(0);
  auto kW = INT_ARG(1);
  auto sH = INT_ARG(2);
  auto sW = INT_ARG(3);
  auto pH = INT_ARG(4);
  auto pW = INT_ARG(5);
  auto dH = INT_ARG(6);
  auto dW = INT_ARG(7);
  // sd_printf("%s %d\n",__FILE__,__LINE__);
  auto isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;  // INT_ARG(10): 1-NHWC, 0-NCHW

  std::unique_ptr<NDArray> inTemp, outTemp;
  NDArray *in = input, *out = output;
  if (!isNCHW) {
    inTemp.reset(new NDArray(input->permute({0, 3, 1, 2}).dup('c')));
    in = inTemp.get();
    outTemp.reset(new NDArray(output->permute({0, 3, 1, 2}).ulike()));
    out = outTemp.get();
  }

  vednnTensorParam_t ParamIn;
  vednnTensorParam_t ParamOut;

  vednnPoolingParam_t ParamConv;

  ParamIn.dtype = DTYPE_FLOAT;
  ParamIn.batch = (int)in->sizeAt(0);
  ParamIn.channel = (int)in->sizeAt(1);
  ParamIn.height = (int)in->sizeAt(2);
  ParamIn.width = (int)in->sizeAt(3);

  ParamOut.dtype = DTYPE_FLOAT;
  ParamOut.batch = (int)out->sizeAt(0);
  ParamOut.channel = (int)out->sizeAt(1);
  ParamOut.height = (int)out->sizeAt(2);
  ParamOut.width = (int)out->sizeAt(3);

  ParamConv.windowWidth = kW;
  ParamConv.windowHeight = kH;
  ParamConv.strideWidth = sW;
  ParamConv.strideHeight = sH;
  ParamConv.padWidth = pW;
  ParamConv.padHeight = pH;

  vednnError_t res = vednnMaxPoolingForward(&ParamIn, in->buffer(), &ParamOut, out->buffer(), &ParamConv);

  if (out != nullptr && out != output) {
    output->assign(out->permute({0, 2, 3, 1}));
  }
  return res == VEDNN_SUCCESS ? sd::Status::OK : sd::Status::BAD_ARGUMENTS;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(maxpool2d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  auto dH = INT_ARG(6);
  auto dW = INT_ARG(7);
  auto paddingMode = INT_ARG(8);

  Requirements req("VEDNN MAXPOOL2d OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dH, "dilation#H"), 1) && req.expectEq(makeInfoVariable(dW, "dilation#W"), 1) &&
      req.expectEq(makeInfoVariable(paddingMode, "paddingMode"), 0) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 4) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), 4) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->ews(), EWS_MSG_INPUT), 1) &&
      req.expectEq(makeInfoVariable(output->ews(), EWS_MSG_OUTPUT), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
