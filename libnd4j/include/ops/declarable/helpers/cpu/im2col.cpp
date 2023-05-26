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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.09.2018
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/im2col.h>
#include <indexing/NDIndexUtils.h>
#include <ops/declarable/CustomOperations.h>

#if NOT_EXCLUDED(OP_im2col)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void im2col_(sd::LaunchContext& context, const NDArray& input, NDArray& output, const LongType kH, const LongType kW,
                    const LongType sH, const LongType sW, const LongType pH, const LongType pW, const LongType dH, const LongType dW,
                    const NDArray& arrZeroPadVal) {
  // input [bS, iC, iH, iW] is convoluted to output [bS, iC, kH, kW, oH, oW]

  auto imBuff = static_cast<T const*>(input.buffer());
  auto colBuff = static_cast<T*>(output.buffer());
  auto imShapeBuffer = input.shapeInfo();
  auto colShapeBuffer = output.shapeInfo();
  auto colShape = shape::shapeOf(colShapeBuffer);
  auto colStride = shape::stride(colShapeBuffer);
  auto imShape = shape::shapeOf(imShapeBuffer);
  auto imStride = shape::stride(imShapeBuffer);

  const T zeroPadVal = arrZeroPadVal.e<T>(0);

  const LongType bS = imShape[0];
  const LongType iC = imShape[1];
  const LongType iH = imShape[2];
  const LongType iW = imShape[3];
  const LongType oH = colShape[4];
  const LongType oW = colShape[5];
  const sd::LongType colStride0 = colStride[0];
  const sd::LongType colStride1 = colStride[1];
  const sd::LongType colStride2 = colStride[2];
  const sd::LongType colStride3 = colStride[3];
  const sd::LongType colStride4 = colStride[4];
  const sd::LongType colStride5 = colStride[5];
  const sd::LongType imStride0 = imStride[0];
  const sd::LongType imStride1 = imStride[1];
  const sd::LongType imStride2 = imStride[2];
  const sd::LongType imStride3 = imStride[3];
  sd::ops::create_view createView;
  auto all = sd::NDIndexUtils::createAll();

  auto recastInput = reinterpret_cast<NDArray *>(const_cast<NDArray *>(&input));
  auto recastOutput = reinterpret_cast<NDArray *>(const_cast<NDArray *>(&output));
  sd::ops::pad pad2;
  const std::vector<sd::LongType> values = {0,0,0,0,pH,pH + sH - 1,pW,pW + sW - 1};
  const std::vector<sd::LongType>  shape = {4,2};

  auto inputPad = NDArrayFactory::create('c',shape,values).cast(sd::DataType::INT32);
  auto padded = pad2.evaluate({recastInput,&inputPad},{},{0}, {zeroPadVal}).at(0);
  auto retGet = NDArrayFactory::create(input.dataType(),{output.sizeAt(0),output.sizeAt(1),kH,kW,oH,oW},input.dataType());
  for(int i = 0; i < kH; i++) {
    sd::LongType iLim = i + sH * oH;
    for(int j = 0; j < kW; j++) {
      sd::LongType jLim = j + sW * oW;
      auto interval = sd::NDIndexUtils::createInterval(i,iLim,sH,0);
      auto interval2 = sd::NDIndexUtils::createInterval(j,jLim,sW,0);
      auto point = sd::NDIndexUtils::createPoint(i);
      auto point2 = sd::NDIndexUtils::createPoint(j);
      sd_printf("About to do evaluate for interval 1: i %d iLim %d sH %d inclusive %d\n",i,iLim,sH,0);
      sd_printf("About to do evaluate for interval 2: j %d jLim %d sW %d inclusive %d\n",j,jLim,sW,0);
      padded->printShapeInfo("Interval source");
      auto get = createView.evaluate({padded, &all, &all, &interval, &interval2}).at(0);
      sd_printf("About to do evaluate for point %d\n",0);
      sd_printf("About to do evaluate for point %d\n",i);
      sd_printf("About to do evaluate for point %d\n",j);
      retGet.printShapeInfo("Point array source");
      auto assignView = createView.evaluate({&retGet,&all, &all,&point, &point2, &all, &all}).at(0);
      sd_printf("get shape %s\n",ShapeUtils::shapeAsString(get->shapeInfo()).c_str());
      sd_printf("assign shape %s\n",ShapeUtils::shapeAsString(assignView->shapeInfo()).c_str());
      assignView->assign(get);

    }
  }

  output.assign(retGet.reshape('c',output.getShapeAsVector()));
}

void im2col(sd::LaunchContext& context, const NDArray& im, NDArray& col, const LongType kH, const LongType kW, const LongType sH,
            const LongType sW, const LongType pH, const LongType pW, const LongType dH, const LongType dW, const NDArray& arrZeroPadVal) {
#if defined(HAVE_VEDA)
  NDArray::preparePrimaryUse({&col}, {&im});
#endif
  BUILD_SINGLE_SELECTOR(im.dataType(), im2col_, (context, im, col, kH, kW, sH, sW, pH, pW, dH, dW, arrZeroPadVal),
                        SD_FLOAT_TYPES);
#if defined(HAVE_VEDA)
  NDArray::registerPrimaryUse({&col}, {&im});
#endif
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif