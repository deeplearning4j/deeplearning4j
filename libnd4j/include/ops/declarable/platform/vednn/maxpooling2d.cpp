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

  auto isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;  // INT_ARG(10): 1-NHWC, 0-NCHW

  std::unique_ptr<NDArray> inTemp, outTemp;
  NDArray *in = input, *out = output;
  if (!isNCHW) {
    inTemp.reset(new NDArray(input->permute({0, 3, 1, 2}).dup('c')));
    in = inTemp.get();
    outTemp.reset(new NDArray(output->permute({0, 3, 1, 2}).ulike()));
    out = outTemp.get();
  }

  vednnTensorParam_t paramIn;
  vednnTensorParam_t paramOut;

  vednnPoolingParam_t paramConv;

  paramIn.dtype = DTYPE_FLOAT;
  paramIn.batch = (int)in->sizeAt(0);
  paramIn.channel = (int)in->sizeAt(1);
  paramIn.height = (int)in->sizeAt(2);
  paramIn.width = (int)in->sizeAt(3);

  paramOut.dtype = DTYPE_FLOAT;
  paramOut.batch = (int)out->sizeAt(0);
  paramOut.channel = (int)out->sizeAt(1);
  paramOut.height = (int)out->sizeAt(2);
  paramOut.width = (int)out->sizeAt(3);

  paramConv.windowWidth = kW;
  paramConv.windowHeight = kH;
  paramConv.strideWidth = sW;
  paramConv.strideHeight = sH;
  paramConv.padWidth = pW;
  paramConv.padHeight = pH;

  vednnError_t res = vednnMaxPoolingForward(&paramIn, in->buffer(), &paramOut, out->buffer(), &paramConv);

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
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dH, "dilation#H"), 1) && req.expectEq(makeInfoVariable(dW, "dilation#W"), 1) &&
      req.expectEq(makeInfoVariable(paddingMode, "paddingMode"), 0) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
      req.expectEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), 4) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->ews(), EWS_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(input->ews(), EWS_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(output->ews(), EWS_MSG_OUTPUT), 1);
  req.logTheSuccess();
  return req;
}

PLATFORM_IMPL(maxpool2d_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto gradOut = INPUT_VARIABLE(1); 
  auto gradIn = OUTPUT_VARIABLE(0);
  auto kH = INT_ARG(0);
  auto kW = INT_ARG(1);
  auto sH = INT_ARG(2);
  auto sW = INT_ARG(3);
  auto pH = INT_ARG(4);
  auto pW = INT_ARG(5);
  auto dH = INT_ARG(6);
  auto dW = INT_ARG(7);

  auto isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;  // INT_ARG(10): 1-NHWC, 0-NCHW
  std::unique_ptr<NDArray> inTemp, gradOutTemp, gradInTemp;
  
  NDArray *in = input, *gradOutPtr = gradOut, *gradInPtr = gradIn, *out;
  if (!isNCHW) {
    inTemp.reset(new NDArray(input->permute({0, 3, 1, 2}).dup('c')));
    in = inTemp.get();
    gradOutTemp.reset(new NDArray(gradOut->permute({0, 3, 1, 2}).dup('c')));
    gradOutPtr = gradOutTemp.get();
    gradInTemp.reset(new NDArray(gradIn->permute({0, 3, 1, 2}).ulike()));
    gradInPtr = gradInTemp.get();
  }
  NDArray output = gradOutPtr->ulike();
  out = &output;
  vednnTensorParam_t paramIn, paramGradOut, paramGradIn, paramOut;
  vednnPoolingParam_t paramConv;
  paramIn.dtype = DTYPE_FLOAT;
  paramIn.batch = (int)in->sizeAt(0);
  paramIn.channel = (int)in->sizeAt(1);
  paramIn.height = (int)in->sizeAt(2);
  paramIn.width = (int)in->sizeAt(3);

  paramGradOut.dtype = DTYPE_FLOAT;
  paramGradOut.batch = (int)gradOutPtr->sizeAt(0);
  paramGradOut.channel = (int)gradOutPtr->sizeAt(1);
  paramGradOut.height = (int)gradOutPtr->sizeAt(2);
  paramGradOut.width = (int)gradOutPtr->sizeAt(3);

  paramGradIn.dtype = DTYPE_FLOAT;
  paramGradIn.batch = (int)gradInPtr->sizeAt(0);
  paramGradIn.channel = (int)gradInPtr->sizeAt(1);
  paramGradIn.height = (int)gradInPtr->sizeAt(2);
  paramGradIn.width = (int)gradInPtr->sizeAt(3);

  paramOut.dtype = DTYPE_FLOAT;
  paramOut.batch = (int)out->sizeAt(0);
  paramOut.channel = (int)out->sizeAt(1);
  paramOut.height = (int)out->sizeAt(2);
  paramOut.width = (int)out->sizeAt(3);

  paramConv.windowWidth = kW;
  paramConv.windowHeight = kH;
  paramConv.strideWidth = sW;
  paramConv.strideHeight = sH;
  paramConv.padWidth = pW;
  paramConv.padHeight = pH;
  vednnError_t res = vednnMaxPoolingForward(&paramIn, in->buffer(), &paramOut, out->buffer(), &paramConv);

  if(res != VEDNN_SUCCESS) return sd::Status::BAD_ARGUMENTS;
  res = vednnMaxPoolingBackward(&paramGradOut, gradOutPtr->buffer(),  &paramOut, out->buffer(), &paramIn, in->buffer(), &paramGradIn, gradInPtr->buffer(), &paramConv);

  if (gradIn != nullptr && gradInPtr != gradIn) {
    gradIn->assign(gradInPtr->permute({0, 2, 3, 1}));
  }

  return res == VEDNN_SUCCESS ? sd::Status::OK : sd::Status::BAD_ARGUMENTS;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(maxpool2d_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto gradOut = INPUT_VARIABLE(1); 
  auto output = OUTPUT_VARIABLE(0);
  auto dH = INT_ARG(6);
  auto dW = INT_ARG(7);
  auto paddingMode = INT_ARG(8);

  Requirements req("VEDNN MAXPOOL2d OP");
#if 1
  sd::Environment::getInstance().setDebug(true) ;
  sd::Environment::getInstance().setVerbose(true);
#endif
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gradOut->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dH, "dilation#H"), 1) && req.expectEq(makeInfoVariable(dW, "dilation#W"), 1) &&
      req.expectEq(makeInfoVariable(paddingMode, "paddingMode"), 0) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
      req.expectEq(makeInfoVariable(gradOut->rankOf(), RANK_MSG_INPUT1), 4) &&
      req.expectEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), 4) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(gradOut->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->ews(), EWS_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(gradOut->ews(), EWS_MSG_INPUT1), 1) &&
      req.expectEq(makeInfoVariable(output->ews(), EWS_MSG_OUTPUT), 1);
  req.logTheSuccess();

  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
