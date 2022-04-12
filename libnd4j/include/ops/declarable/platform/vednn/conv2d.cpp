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

std::unique_ptr<NDArray> newWeight_3x3(const NDArray &w, int weightFormat) {
  sd::LongType oC, iC, kH, kW, oStride2, iStride2, hStride2, wStride2;

  // [kH, kW, iC, oC] -> [oC, iC, kH, kW]
  oC = w.sizeAt(3);
  iC = w.sizeAt(2);
  kH = w.sizeAt(0);
  kW = w.sizeAt(1);
  assert(kH == 3 && kW == 3);
  oStride2 = w.strideAt(3);
  iStride2 = w.strideAt(2);
  hStride2 = w.strideAt(0);
  wStride2 = w.strideAt(1);
  auto context = w.getContext();
  std::vector<sd::LongType> shape = {oC, iC, kH, kW};
  // DataType type, const char order, const std::vector<sd::LongType> &shape
  ShapeDescriptor shapeDescriptor(w.dataType(), 'c', shape);
  sd::LongType allocSize = shapeDescriptor.allocLength() * DataTypeUtils::sizeOfElement(shapeDescriptor.dataType());
  std::shared_ptr<DataBuffer> buffer =
      std::make_shared<DataBuffer>(allocSize, shapeDescriptor.dataType(), context->getWorkspace());

  std::unique_ptr<NDArray> arr(new NDArray(buffer, shapeDescriptor, context));
  auto oStride1 = arr->strideAt(0);
  auto iStride1 = arr->strideAt(1);
  auto hStride1 = arr->strideAt(2);

  auto bIn = w.bufferAsT<float>();
  auto bOut = arr->bufferAsT<float>();
  auto bIn_0 = bIn;
  auto bIn_1 = bIn + wStride2;
  auto bIn_2 = bIn + wStride2 + wStride2;

  auto bIn1_0 = bIn_0 + hStride2;
  auto bIn1_1 = bIn_1 + hStride2;
  auto bIn1_2 = bIn_2 + hStride2;

  auto bIn2_0 = bIn1_0 + hStride2;
  auto bIn2_1 = bIn1_1 + hStride2;
  auto bIn2_2 = bIn1_2 + hStride2;

  auto bOut_0 = bOut;
  auto bOut_1 = bOut + 1;
  auto bOut_2 = bOut + 2;

  auto bOut1_0 = bOut_0 + hStride1;
  auto bOut1_1 = bOut_1 + hStride1;
  auto bOut1_2 = bOut_2 + hStride1;

  auto bOut2_0 = bOut1_0 + hStride1;
  auto bOut2_1 = bOut1_1 + hStride1;
  auto bOut2_2 = bOut1_2 + hStride1;
// float
#pragma omp parallel for
  for (int j = 0; j < iC; j++) {
    for (int i = 0; i < oC; i++) {
      bOut_0[i * oStride1 + j * iStride1] = bIn_0[i + j * iStride2];
      bOut_1[i * oStride1 + j * iStride1] = bIn_1[i + j * iStride2];
      bOut_2[i * oStride1 + j * iStride1] = bIn_2[i + j * iStride2];
      bOut1_0[i * oStride1 + j * iStride1] = bIn1_0[i + j * iStride2];
      bOut1_1[i * oStride1 + j * iStride1] = bIn1_1[i + j * iStride2];
      bOut1_2[i * oStride1 + j * iStride1] = bIn1_2[i + j * iStride2];
      bOut2_0[i * oStride1 + j * iStride1] = bIn2_0[i + j * iStride2];
      bOut2_1[i * oStride1 + j * iStride1] = bIn2_1[i + j * iStride2];
      bOut2_2[i * oStride1 + j * iStride1] = bIn2_2[i + j * iStride2];
    }
  }

  return arr;
}

//////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv2d, ENGINE_CPU) {

  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]

  auto output = OUTPUT_VARIABLE(0);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

  int sH = INT_ARG(2);           // strides height
  int sW = INT_ARG(3);           // strides width
  int pH = INT_ARG(4);           // paddings height
  int pW = INT_ARG(5);           // paddings width
  int dH = INT_ARG(6);           // dilations height
  int dW = INT_ARG(7);           // dilations width
  int paddingMode = INT_ARG(8);  // 0-VALID, 1-SAME
  // INT_ARG(9): 0-NCHW,  1-NHWC
  bool isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;
  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]
  int weightFormat = block.getIArguments()->size() > 10 ? INT_ARG(10) : 0;

  int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));  // filter(kernel) height
  int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));  // filter(kernel) width

  // batch size, input channels, input height/width, output channels, output height/width;
  int bS, iC, iH, iW, oC, oH, oW;
  int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                             indWiC, indWoC, indWkH, indOoH);

  ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

  // int pWSame = (paddingMode == 2 && dW > 1) ? ((oW - 1) * sW + (kW - 1) * dW + 1 - iW) / 2
  //                                           : pW;  // dH == 1 for causal mode in conv1d
  // int padLeft = pW;
  // int padTop = pH;
  // int padRight = (oW - 1) * sW - iW + kW - pWSame;
  // int padBottom = (oH - 1) * sH - iH + kH - pH;

  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(weightFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CONV2D VEDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CONV2D VEDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, "
                 "%i instead !",
                 oC, bias->rankOf(), bias->lengthOf());

  vednnTensorParam_t paramIn;
  vednnBiasParam_t paramBias;
  vednnFilterParam_t paramFilter;
  vednnTensorParam_t paramOut;

  vednnConvolutionParam_t paramConv;
  NDArray *w = weights, *in = input, *out = output;

#if !defined(HAVE_VEDA)
  std::unique_ptr<NDArray> wTemp, inTemp, outTemp;

  if (0 == weightFormat) {
    // [kH, kW, iC, oC] -> [oC, iC, kH, kW]
    if (weights->ordering() == 'c' && weights->ews() == 1 && weights->sizeAt(0) == 3 && weights->sizeAt(1) == 3) {
      wTemp = newWeight_3x3(*weights, weightFormat);
    } else {
      wTemp.reset(new NDArray(weights->permute({3, 2, 0, 1}).dup('c')));
    }
    w = wTemp.get();

  } else if (2 == weightFormat) {
    // [oC, kH, kW, iC] -> [oC, iC, kH, kW]
    wTemp.reset(new NDArray(weights->permute({0, 3, 1, 2}).dup('c')));
    w = wTemp.get();
  }

  if (!isNCHW) {
    inTemp.reset(new NDArray(input->permute({0, 3, 1, 2}).dup('c')));
    in = inTemp.get();
    outTemp.reset(new NDArray(output->permute({0, 3, 1, 2}).ulike()));
    out = outTemp.get();
  }
#endif

  if (bias) {
    paramBias.dtype = DTYPE_FLOAT;
    paramBias.channel = bias->lengthOf();
  }

  paramIn = getTensorFormat(*in, isNCHW);
  //// 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]
  paramFilter = getFilterParam(*w, weightFormat);

  paramOut = getTensorFormat(*out, isNCHW);

  paramConv.group = 1;
  paramConv.strideWidth = sW;     // col stride    W
  paramConv.strideHeight = sH;    // row stride    H
  paramConv.dilationWidth = dW;   // col dilation  W
  paramConv.dilationHeight = dH;  // row dilation  H
  paramConv.padWidth = pW;        // col padding   W
  paramConv.padHeight = pH;       // row padding   H

#if !defined(HAVE_VEDA)

  vednnError_t res;
  if (bias) {
    res = vednnConvolutionForwardAddBias(&paramIn, in->buffer(), &paramFilter, w->buffer(), &paramBias, bias->buffer(),
                                         &paramOut, out->buffer(), &paramConv, VEDNN_CONV_ALGORITHM_DIRECT);
  } else {
    res = vednnConvolutionForward(&paramIn, in->buffer(), &paramFilter, w->buffer(), &paramOut, out->buffer(),
                                  &paramConv, VEDNN_CONV_ALGORITHM_DIRECT);
  }

  auto status = res == VEDNN_SUCCESS ? sd::Status::OK : sd::Status::BAD_ARGUMENTS;
#else

  VEDA_HANDLE &handle = VEDA::getInstance().getVEDA_HANDLE(0);
  SCOPED_VEDA_CONTEXT scopedContext(handle.getDevice());

  auto func = handle.getFunctionByConstPtrName("vedaVednnConvolutionForwardAddBias");

  VEDAdeviceptr vIn, vW, vO;
  VEDAdeviceptr vB = nullptr;
  size_t sizeIn = in->lengthOf() * in->sizeOfT();
  size_t sizeW = w->lengthOf() * w->sizeOfT();
  size_t sizeB = bias ? bias->lengthOf() * bias->sizeOfT() : 0;
  size_t sizeO = out->lengthOf() * out->sizeOfT();

  VEDA_CALL_THROW(vedaMemAllocAsync(&vIn, sizeIn, 0));
  VEDA_CALL_THROW(vedaMemAllocAsync(&vW, sizeW, 0));
  if (bias) VEDA_CALL_THROW(vedaMemAllocAsync(&vB, sizeB, 0));
  VEDA_CALL_THROW(vedaMemAllocAsync(&vO, sizeO, 0));

  VEDA_CALL_THROW(vedaMemcpyHtoDAsync(vIn, in->buffer(), sizeIn, 0));
  VEDA_CALL_THROW(vedaMemcpyHtoDAsync(vW, w->buffer(), sizeW, 0));
  if (bias) VEDA_CALL_THROW(vedaMemcpyHtoDAsync(vB, bias->buffer(), sizeB, 0));

  // if(bias) sd_printf("%s\n","--------bias case--------");

  VEDA_CALL_THROW(vedaLaunchKernel(func, 0, VEDAstack(&paramIn, VEDA_ARGS_INTENT_IN, sizeof(paramIn)), vIn, (uint8_t)isNCHW,
                             VEDAstack(&paramFilter, VEDA_ARGS_INTENT_IN, sizeof(paramFilter)), vW, (int32_t)weightFormat,
                             VEDAstack(&paramBias, VEDA_ARGS_INTENT_IN, sizeof(paramBias)), vB,
                             VEDAstack(&paramOut, VEDA_ARGS_INTENT_IN, sizeof(paramOut)), vO,  (uint8_t)isNCHW,
                             VEDAstack(&paramConv, VEDA_ARGS_INTENT_IN, sizeof(paramConv)),
                             (int)VEDNN_CONV_ALGORITHM_DIRECT));

  VEDA_CALL_THROW(vedaMemcpyDtoHAsync(out->buffer(), vO, sizeO, 0));

  VEDA_CALL_THROW(vedaMemFreeAsync(vIn, 0));
  VEDA_CALL_THROW(vedaMemFreeAsync(vW, 0));
  if (bias) VEDA_CALL_THROW(vedaMemFreeAsync(vB, 0));
  VEDA_CALL_THROW(vedaMemFreeAsync(vO, 0));
  scopedContext.sync();

  auto status = sd::Status::OK;
#endif

#if !defined(HAVE_VEDA)
  if (out != nullptr && out != output) {
    output->assign(out->permute({0, 2, 3, 1}));
  }
#endif
  return status;
}

PLATFORM_CHECK(conv2d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
  auto output = OUTPUT_VARIABLE(0);
  auto paddingMode = INT_ARG(8);

  Requirements req("VEDNN CONV2d OP");
  // Note: For kW,kH==2 and paddingMode = 1 (same) Vednn was failing to output correct results
  // So we decided to restrict it
  req.expectEq(makeInfoVariable(paddingMode, "paddingMode"), 0) &&
      // input related constraints
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(input->ews(), EWS_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(weights->rankOf(), RANK_MSG_INPUT1), 4) &&
      req.expectEq(makeInfoVariable(weights->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectEq(makeInfoVariable(weights->ews(), EWS_MSG_INPUT1), 1) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), 4) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ews(), EWS_MSG_OUTPUT), 1);
  if (bias) {
    req.expectEq(makeInfoVariable(bias->dataType(), TYPE_MSG_INPUT2), DataType::FLOAT32) &&
        req.expectEq(makeInfoVariable(bias->ordering(), ORDERING_MSG_INPUT2), 'c') &&
        req.expectEq(makeInfoVariable(bias->ews(), EWS_MSG_INPUT2), 1);
  }
  req.logTheSuccess();
  return req;
}

PLATFORM_IMPL(conv2d_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);  // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
  auto gradW = OUTPUT_VARIABLE(1);  // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;
  int kH = INT_ARG(0);                                               // filter(kernel) height
  int kW = INT_ARG(1);                                               // filter(kernel) width
  int sH = INT_ARG(2);                                               // strides height
  int sW = INT_ARG(3);                                               // strides width
  int pH = INT_ARG(4);                                               // paddings height
  int pW = INT_ARG(5);                                               // paddings width
  int dH = INT_ARG(6);                                               // dilations height
  int dW = INT_ARG(7);                                               // dilations width
  int paddingMode = INT_ARG(8);                                      // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW, 1-NHWC
  int weightFormat = block.getIArguments()->size() > 10
                         ? INT_ARG(10)
                         : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  int bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, weightFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  int trueoH, trueoW;  // true output height, width
  ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, paddingMode);

  if (paddingMode)  // SAME
    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);

  std::vector<sd::LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, oC, trueoH, trueoW, 0, indIOioC, indOoH, indOoH + 1});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(weightFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(
      gradO->isSameShape(expectedGradOShape), 0,
      "CONV2D_BP VEDNN OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CONV2D_BP VEDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());

  vednnTensorParam_t paramIn, paramGradOut, paramGradIn;
  vednnFilterParam_t paramFilter;
  vednnConvolutionParam_t paramConv;

  std::unique_ptr<NDArray> inTemp, wTemp, gradOutTemp, gradInTemp, gradWeightsTemp;
  NDArray *in = input, *weightPtr = weights, *gradOutPtr = gradO, *gradInPtr = gradI, *gradWeightsPtr = gradW;
  if (0 == weightFormat) {
    // [kH, kW, iC, oC] -> [oC, iC, kH, kW]
    if (weights->ordering() == 'c' && weights->ews() == 1 && weights->sizeAt(0) == 3 && weights->sizeAt(1) == 3) {
      wTemp = newWeight_3x3(*weights, weightFormat);
    } else {
      wTemp.reset(new NDArray(weights->permute({3, 2, 0, 1}).dup('c')));
    }
    weightPtr = wTemp.get();
  } else if (2 == weightFormat) {
    // [oC, kH, kW, iC] -> [oC, iC, kH, kW]
    wTemp.reset(new NDArray(weights->permute({0, 3, 1, 2}).dup('c')));
    weightPtr = wTemp.get();
  }
  if (weightPtr != weights) {
    gradWeightsTemp.reset(new NDArray(weightPtr->ulike()));
    gradWeightsPtr = gradWeightsTemp.get();
  }
  if (!isNCHW) {
    inTemp.reset(new NDArray(input->permute({0, 3, 1, 2}).dup('c')));
    in = inTemp.get();
    gradOutTemp.reset(new NDArray(gradO->permute({0, 3, 1, 2}).dup('c')));
    gradOutPtr = gradOutTemp.get();
    gradInTemp.reset(new NDArray(gradI->permute({0, 3, 1, 2}).ulike()));
    gradInPtr = gradInTemp.get();
  }

  paramGradOut = getTensorFormat(*gradOutPtr);

  paramFilter = getFilterParam(*weightPtr, 1);

  paramGradIn = getTensorFormat(*gradInPtr);

  paramConv.group = 1;
  paramConv.strideWidth = sW;     // col stride    W
  paramConv.strideHeight = sH;    // row stride    H
  paramConv.dilationWidth = dW;   // col dilation  W
  paramConv.dilationHeight = dH;  // row dilation  H
  paramConv.padWidth = pW;        // col padding   W
  paramConv.padHeight = pH;       // row padding   H
#if !defined(HAVE_VEDA)
  vednnError_t resData =
      vednnConvolutionBackwardData(&paramGradOut, gradOutPtr->buffer(), &paramFilter, weightPtr->buffer(), &paramGradIn,
                                   gradInPtr->buffer(), &paramConv, VEDNN_CONV_ALGORITHM_DIRECT);

  // paramGradIn could be used for "in"
  // paramFilter could be used for "gradWeightsPtr"
  vednnError_t resFilter =
      vednnConvolutionBackwardFilter(&paramGradIn, in->buffer(), &paramGradOut, gradOutPtr->buffer(), &paramFilter,
                                     gradWeightsPtr->buffer(), &paramConv, VEDNN_CONV_ALGORITHM_DIRECT);
  auto status = (resData == VEDNN_SUCCESS && resFilter == VEDNN_SUCCESS) ? sd::Status::OK : sd::Status::BAD_ARGUMENTS;
#else
  VEDA_HANDLE &handle = VEDA::getInstance().getVEDA_HANDLE(0);
  SCOPED_VEDA_CONTEXT scopedContext(handle.getDevice());

  auto func = handle.getFunctionByConstPtrName("vedaVednnConvolutionBackwardDataAndFilter");
  VEDAdeviceptr vGradOut, vW, vGradW, vIn, vGradIn;

  VEDA_CALL_THROW(vedaMemAllocAsync(&vGradOut, gradOutPtr->lengthOf() * gradOutPtr->sizeOfT(), 0));
  VEDA_CALL_THROW(vedaMemAllocAsync(&vW, weightPtr->lengthOf() * weightPtr->sizeOfT(), 0));
  VEDA_CALL_THROW(vedaMemAllocAsync(&vGradW, gradWeightsPtr->lengthOf() * gradWeightsPtr->sizeOfT(), 0));
  VEDA_CALL_THROW(vedaMemAllocAsync(&vIn, in->lengthOf() * in->sizeOfT(), 0));
  VEDA_CALL_THROW(vedaMemAllocAsync(&vGradIn, gradInPtr->lengthOf() * gradInPtr->sizeOfT(), 0));

  VEDA_CALL_THROW(vedaMemcpyHtoDAsync(vGradOut, gradOutPtr->buffer(), gradOutPtr->lengthOf() * gradOutPtr->sizeOfT(), 0));
  VEDA_CALL_THROW(vedaMemcpyHtoDAsync(vW, weightPtr->buffer(), weightPtr->lengthOf() * weightPtr->sizeOfT(), 0));
  // VEDA_CALL_THROW(
  //     vedaMemcpyHtoDAsync(vGradW, gradWeightsPtr->buffer(), gradWeightsPtr->lengthOf() * gradWeightsPtr->sizeOfT(),
  //     0));
  VEDA_CALL_THROW(vedaMemcpyHtoDAsync(vIn, in->buffer(), in->lengthOf() * in->sizeOfT(), 0));
  // VEDA_CALL_THROW(vedaMemcpyHtoDAsync(vGradIn, gradInPtr->buffer(), gradInPtr->lengthOf() * gradInPtr->sizeOfT(), 0));

  VEDA_CALL_THROW(vedaLaunchKernel(func, 0, VEDAstack(&paramGradOut, VEDA_ARGS_INTENT_IN, sizeof(paramGradOut)), vGradOut,
                             VEDAstack(&paramFilter, VEDA_ARGS_INTENT_IN, sizeof(paramFilter)), vW, vGradW,
                             VEDAstack(&paramGradIn, VEDA_ARGS_INTENT_IN, sizeof(paramGradIn)), vIn, vGradIn,
                             VEDAstack(&paramConv, VEDA_ARGS_INTENT_IN, sizeof(paramConv)),
                             VEDNN_CONV_ALGORITHM_DIRECT));

  VEDA_CALL_THROW(vedaMemcpyDtoHAsync(gradInPtr->buffer(), vGradIn, gradInPtr->lengthOf() * gradInPtr->sizeOfT(), 0));
  VEDA_CALL_THROW(
      vedaMemcpyDtoHAsync(gradWeightsPtr->buffer(), vGradW, gradWeightsPtr->lengthOf() * gradWeightsPtr->sizeOfT(), 0));

  VEDA_CALL_THROW(vedaMemFreeAsync(vGradOut, 0));
  VEDA_CALL_THROW(vedaMemFreeAsync(vW, 0));
  VEDA_CALL_THROW(vedaMemFreeAsync(vGradW, 0));
  VEDA_CALL_THROW(vedaMemFreeAsync(vIn, 0));
  VEDA_CALL_THROW(vedaMemFreeAsync(vGradIn, 0));

  scopedContext.sync();

  // TODO: after replacing it with vedaLaunchKernelEx we will return result based on vednn veda call
  auto status = sd::Status::OK;
#endif
  if (gradInPtr != nullptr && gradInPtr != gradI) {
    gradI->assign(gradInPtr->permute({0, 2, 3, 1}));
  }
  if (gradWeightsPtr != nullptr && gradWeightsPtr != gradW) {
    // [oC, iC, kH, kW] -> [kH, kW, iC, oC]
    if (weightFormat == 0) gradW->assign(gradWeightsPtr->permute({2, 3, 1, 0}));
    // [oC, iC, kH, kW] -> [oC, kH, kW, iC]
    else
      gradW->assign(gradWeightsPtr->permute({0, 2, 3, 1}));
  }
  // we calculate bias ourselves
  if (gradB) {
    std::vector<int> gradOaxesForDot;
    if (!isNCHW) {
      gradOaxesForDot = {0, 1, 2};
    } else {
      gradOaxesForDot = {0, 2, 3};  // bS, oH, oW
    }
    NDArray *gradBiasPtr = gradB;
    std::unique_ptr<NDArray> gradBiasTemp;
    if (gradB->rankOf() == 2) {
      gradBiasTemp.reset(new NDArray(gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()})));
      gradBiasPtr = gradBiasTemp.get();
    }
    gradO->reduceAlongDimension(reduce::Sum, *gradBiasPtr, gradOaxesForDot, false);  // sum over bS, oH, oW
  }
  return status;
}

PLATFORM_CHECK(conv2d_bp, ENGINE_CPU) {
  int paddingMode = INT_ARG(8);
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;
  auto gradO = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);

  auto gradI = OUTPUT_VARIABLE(0);
  auto gradW = OUTPUT_VARIABLE(1);
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;

  Requirements req("VEDNN CONV2d BP OP");
  req.expectEq(makeInfoVariable(paddingMode, "paddingMode"), 0) &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(input->ews(), EWS_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(weights->rankOf(), RANK_MSG_INPUT1), 4) &&
      req.expectEq(makeInfoVariable(weights->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectEq(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT2), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gradO->rankOf(), RANK_MSG_INPUT2), 4) &&
      req.expectEq(makeInfoVariable(gradO->ordering(), ORDERING_MSG_INPUT2), 'c') &&
      req.expectEq(makeInfoVariable(gradO->ews(), EWS_MSG_INPUT2), 1);
  req.expectEq(makeInfoVariable(gradI->dataType(), TYPE_MSG_OUTPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gradI->rankOf(), RANK_MSG_OUTPUT0), 4) &&
      req.expectEq(makeInfoVariable(gradI->ordering(), ORDERING_MSG_OUTPUT0), 'c') &&
      req.expectEq(makeInfoVariable(gradI->ews(), EWS_MSG_OUTPUT0), 1) &&
      req.expectEq(makeInfoVariable(gradW->dataType(), TYPE_MSG_OUTPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gradW->rankOf(), RANK_MSG_OUTPUT1), 4) &&
      req.expectEq(makeInfoVariable(gradW->ordering(), ORDERING_MSG_OUTPUT1), 'c');

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
