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

std::unique_ptr<NDArray> newWeight_3x3(const NDArray &w, int wFormat){
    sd::LongType oC, iC, kH, kW, oStride2, iStride2, hStride2, wStride2;

    // [kH, kW, iC, oC] -> [oC, iC, kH, kW]
    oC = w.sizeAt(3);
    iC = w.sizeAt(2);
    kH = w.sizeAt(0);
    kW = w.sizeAt(1); 
    assert(kH==3 && kW==3);
    oStride2 = w.strideAt(3);
    iStride2 = w.strideAt(2);
    hStride2 = w.strideAt(0);
    wStride2 = w.strideAt(1); 
    auto context =  w.getContext();
    std::vector<sd::LongType> shape={oC, iC, kH, kW };
    // DataType type, const char order, const std::vector<sd::LongType> &shape
    ShapeDescriptor shapeDescriptor(w.dataType(), 'c', shape);
      sd::LongType allocSize = shapeDescriptor.allocLength() * DataTypeUtils::sizeOfElement(shapeDescriptor.dataType());
    std::shared_ptr<DataBuffer> buffer =
      std::make_shared<DataBuffer>(allocSize, shapeDescriptor.dataType(), context->getWorkspace());

    std::unique_ptr<NDArray> arr(new NDArray  (buffer, shapeDescriptor,context)); 
    auto oStride1 = arr->strideAt(0);
    auto iStride1 = arr->strideAt(1);
    auto hStride1 = arr->strideAt(2);

    auto bIn = w.bufferAsT<float>();
    auto bOut = arr->bufferAsT<float>();
    auto bIn_0=bIn;
    auto bIn_1=bIn + wStride2;
    auto bIn_2=bIn + wStride2 + wStride2;
    
    auto bIn1_0=bIn_0 + hStride2;
    auto bIn1_1=bIn_1 + hStride2;
    auto bIn1_2=bIn_2 + hStride2;

    auto bIn2_0=bIn1_0 + hStride2;
    auto bIn2_1=bIn1_1 + hStride2;
    auto bIn2_2=bIn1_2 + hStride2;

    auto bOut_0=bOut;
    auto bOut_1=bOut + 1;
    auto bOut_2=bOut + 2;
    
    auto bOut1_0=bOut_0 + hStride1;
    auto bOut1_1=bOut_1 + hStride1;
    auto bOut1_2=bOut_2 + hStride1;

    auto bOut2_0=bOut1_0 + hStride1;
    auto bOut2_1=bOut1_1 + hStride1;
    auto bOut2_2=bOut1_2 + hStride1;
    //float
    #pragma omp parallel for
    for(int j=0;j<iC;j++){
        for(int i=0;i<oC;i++){
        
            bOut_0[i*oStride1 + j* iStride1] = bIn_0[i + j* iStride2];
            bOut_1[i*oStride1 + j* iStride1] = bIn_1[i + j* iStride2];
            bOut_2[i*oStride1 + j* iStride1] = bIn_2[i + j* iStride2];
            bOut1_0[i*oStride1 + j* iStride1] = bIn1_0[i + j* iStride2];
            bOut1_1[i*oStride1 + j* iStride1] = bIn1_1[i + j* iStride2];
            bOut1_2[i*oStride1 + j* iStride1] = bIn1_2[i + j* iStride2];
            bOut2_0[i*oStride1 + j* iStride1] = bIn2_0[i + j* iStride2];
            bOut2_1[i*oStride1 + j* iStride1] = bIn2_1[i + j* iStride2];
            bOut2_2[i*oStride1 + j* iStride1] = bIn2_2[i + j* iStride2];
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

  int sH = INT_ARG(2);                                                // strides height
  int sW = INT_ARG(3);                                                // strides width
  int pH = INT_ARG(4);                                                // paddings height
  int pW = INT_ARG(5);                                                // paddings width
  int dH = INT_ARG(6);                                                // dilations height
  int dW = INT_ARG(7);                                                // dilations width
  int paddingMode = INT_ARG(8);                                       // 0-VALID, 1-SAME
  bool isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 0-NCHW,  1-NHWC
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));  // filter(kernel) height
  int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));  // filter(kernel) width

  // Calculate individual paddings
  unsigned int padLeft, padTop, padRight, padBottom;
  int bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                             indWiC, indWoC, indWkH, indOoH);

  ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW, paddingMode);
  int pWSame = (paddingMode == 2 && dW > 1) ? ((oW - 1) * sW + (kW - 1) * dW + 1 - iW) / 2
                                            : pW;  // dH == 1 for causal mode in conv1d

  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CONV2D VEDNN OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CONV2D VEDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, "
                 "%i instead !",
                 oC, bias->rankOf(), bias->lengthOf());


    vednnTensorParam_t ParamIn ;
    vednnFilterParam_t ParamFilter ;
    vednnTensorParam_t ParamOut ;

    vednnConvolutionParam_t ParamConv ;
    std::unique_ptr<NDArray> wTemp, inTemp, outTemp;
    NDArray *w, *in = input, *out = output;
    if (0 == wFormat){
        // [kH, kW, iC, oC] -> [oC, iC, kH, kW]
        if(weights->ordering()=='c' && weights->ews()==1 && weights->sizeAt(0)==3 && weights->sizeAt(1)==3){
            wTemp = newWeight_3x3(*weights, wFormat);
        }else{
            wTemp.reset(new NDArray(weights->permute( {3, 2, 0, 1} ).dup('c')));
            
        }
        w=wTemp.get(); 
    }
    else if (2 == wFormat){
        // [oC, kH, kW, iC] -> [oC, iC, kH, kW]
        wTemp.reset(new NDArray(weights->permute( {0, 3, 1, 2}).dup('c')));
        w=wTemp.get();
    }
    else{
        w = weights;
    }
    if(!isNCHW){
        inTemp.reset(new NDArray(input->permute( {0, 3, 1, 2} ).dup('c')));
        in=inTemp.get();
        outTemp.reset(new NDArray(output->permute( {0, 3, 1, 2} ).ulike()));
        out=outTemp.get();
    }

    ParamIn.dtype   = DTYPE_FLOAT ;
    ParamIn.batch   = (int)in->sizeAt(0);
    ParamIn.channel = (int)in->sizeAt(1);
    ParamIn.height  = (int)in->sizeAt(2);
    ParamIn.width   = (int)in->sizeAt(3);
    //// 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]
    ParamFilter.dtype      = DTYPE_FLOAT ;
    ParamFilter.inChannel  = (int)in->sizeAt(1);
    ParamFilter.outChannel = (int)out->sizeAt(1);
    ParamFilter.height     = (int)w->sizeAt(2);
    ParamFilter.width      = (int)w->sizeAt(3);
     
    ParamOut.dtype   = DTYPE_FLOAT ;
    ParamOut.batch   = (int)out->sizeAt(0);
    ParamOut.channel = (int)out->sizeAt(1);
    ParamOut.height  = (int)out->sizeAt(2);
    ParamOut.width   = (int)out->sizeAt(3);

    ParamConv.group          = 1 ;
    ParamConv.strideWidth    = sW;  // col stride    W
    ParamConv.strideHeight   = sH;  // row stride    H
    ParamConv.dilationWidth  = dW;  // col dilation  W
    ParamConv.dilationHeight = dH;  // row dilation  H
    ParamConv.padWidth       = pW;  // col padding   W
    ParamConv.padHeight      = pH;  // row padding   H

    vednnConvolutionForward(&ParamIn,     in->buffer(),
                     	    &ParamFilter, w->buffer(),
                     	    &ParamOut,    out->buffer(), 
                     	    &ParamConv,
                     	    VEDNN_CONV_ALGORITHM_DIRECT );

  if(out!=nullptr && out!=output){
      output->assign(out->permute({0,2,3,1}));
  }
  return sd::Status::OK;
}

PLATFORM_CHECK(conv2d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr; 
  auto output = OUTPUT_VARIABLE(0);
  int paddingMode = INT_ARG(8);
  Requirements req("VEDNN CONV2d OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
  req.expectEq(makeInfoVariable(weights->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
  req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
  req.expectEq(bias, (NDArray*)nullptr) &&
  req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4) &&
  req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT0), 'c') &&
  req.expectEq(makeInfoVariable(paddingMode, "paddingMode"), 0) &&
  //req.expectEq(makeInfoVariable(input->stridesOf()[input->rankOf() - 1], "input0#lastStride"), 1) &&
  req.expectEq(makeInfoVariable(weights->rankOf(), RANK_MSG_INPUT1), 4) &&
  req.expectEq(makeInfoVariable(weights->ordering(), ORDERING_MSG_INPUT1), 'c') &&
  //req.expectEq(makeInfoVariable(weights->stridesOf()[weights->rankOf() - 1], "input1#lastStride"), 1) &&
  req.expectEq(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), 4) &&
  req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c');
  //req.expectEq(makeInfoVariable(output->stridesOf()[output->rankOf() - 1], "output#lastStride"), 1);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
