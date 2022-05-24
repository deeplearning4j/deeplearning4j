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

  NDArray *in = input, *out = output;

  vednnTensorParam_t paramIn = getTensorFormat(*in);
  vednnTensorParam_t paramOut = getTensorFormat(*out);

  vednnPoolingParam_t paramConv;

  paramConv.windowWidth = kW;
  paramConv.windowHeight = kH;
  paramConv.strideWidth = sW;
  paramConv.strideHeight = sH;
  paramConv.padWidth = pW;
  paramConv.padHeight = pH;
#if !defined(HAVE_VEDA)
  vednnError_t res = vednnMaxPoolingForward(&paramIn, in->buffer(), &paramOut, out->buffer(), &paramConv);

  auto status = res == VEDNN_SUCCESS ? sd::Status::OK : sd::Status::BAD_ARGUMENTS;
#else

  VEDA_HANDLE& handle = VEDA::getInstance().getVEDA_HANDLE(0);
  auto func = handle.getFunctionByConstPtrName("vedaVednnMaxPoolingForward");
  VEDAdeviceptr vIn, vOut;

  vIn = (VEDAdeviceptr)in->specialBuffer();
  vOut = (VEDAdeviceptr)out->specialBuffer();

  VEDA_CALL_THROW(vedaLaunchKernel(func, 0, VEDAstack(&paramIn, VEDA_ARGS_INTENT_IN, sizeof(paramIn)), vIn,
                                   VEDAstack(&paramOut, VEDA_ARGS_INTENT_IN, sizeof(paramOut)), vOut,

                                   VEDAstack(&paramConv, VEDA_ARGS_INTENT_IN, sizeof(paramConv))));

  auto status = sd::Status::OK;
#endif

  return status;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(maxpool2d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  auto dH = INT_ARG(6);
  auto dW = INT_ARG(7);
  auto paddingMode = INT_ARG(8);

  Requirements req("VEDNN MAXPOOL2d OP");
#if !defined(ALLOW_NHWC_FORMAT)
  auto isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;
  req.expectTrue(makeInfoVariable(isNCHW, "isNCHW")) &&
#endif
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

  NDArray *in = input, *gradOutPtr = gradOut, *gradInPtr = gradIn, *out;

#if !defined(VEDA)
  NDArray output = gradOutPtr->ulike();
  out = &output;
#endif
  vednnTensorParam_t paramIn, paramGradOut, paramGradIn, paramOut;
  vednnPoolingParam_t paramConv;
  paramIn = getTensorFormat(*in);

  paramGradOut = getTensorFormat(*gradOutPtr);

  paramGradIn = getTensorFormat(*gradInPtr);

  paramOut = paramGradOut;

  paramConv.windowWidth = kW;
  paramConv.windowHeight = kH;
  paramConv.strideWidth = sW;
  paramConv.strideHeight = sH;
  paramConv.padWidth = pW;
  paramConv.padHeight = pH;
#if !defined(HAVE_VEDA)
  vednnError_t res = vednnMaxPoolingForward(&paramIn, in->buffer(), &paramOut, out->buffer(), &paramConv);

  if (res != VEDNN_SUCCESS) return sd::Status::BAD_ARGUMENTS;
  res = vednnMaxPoolingBackward(&paramGradOut, gradOutPtr->buffer(), &paramOut, out->buffer(), &paramIn, in->buffer(),
                                &paramGradIn, gradInPtr->buffer(), &paramConv);

  auto status = res == VEDNN_SUCCESS ? sd::Status::OK : sd::Status::BAD_ARGUMENTS;
#else

  VEDA_HANDLE& handle = VEDA::getInstance().getVEDA_HANDLE(0);
  auto func = handle.getFunctionByConstPtrName("vedaVednnMaxPoolingBackwardEx");
  VEDAdeviceptr vGradOut, vOut, vIn, vGradIn;

  vIn = (VEDAdeviceptr)input->specialBuffer();
  vGradOut = (VEDAdeviceptr)gradOutPtr->specialBuffer();
  vGradIn = (VEDAdeviceptr)gradInPtr->specialBuffer();
  // we create temp out and pass it as well
  VEDA_CALL_THROW(vedaMemAllocAsync(&vOut, gradOutPtr->lengthOf() * gradOutPtr->sizeOfT(), 0));
  VEDA_CALL_THROW(vedaLaunchKernel(func, 0, VEDAstack(&paramGradOut, VEDA_ARGS_INTENT_IN, sizeof(paramGradOut)),
                                   vGradOut, VEDAstack(&paramOut, VEDA_ARGS_INTENT_IN, sizeof(paramOut)), vOut,
                                   VEDAstack(&paramIn, VEDA_ARGS_INTENT_IN, sizeof(paramIn)), vIn,
                                   VEDAstack(&paramGradIn, VEDA_ARGS_INTENT_IN, sizeof(paramGradIn)), vGradIn,
                                   VEDAstack(&paramConv, VEDA_ARGS_INTENT_IN, sizeof(paramConv))));

  VEDA_CALL_THROW(vedaMemFreeAsync(vOut, 0));

  auto status = sd::Status::OK;
#endif
  return status;
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
#if !defined(ALLOW_NHWC_FORMAT)
  auto isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;
  req.expectTrue(makeInfoVariable(isNCHW, "isNCHW")) &&
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
