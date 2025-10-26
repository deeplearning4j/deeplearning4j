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
// Created by agibsonccc on 1/19/17.
//
#include <loops/broadcasting.h>

#include "testinclude.h"

class BroadcastMultiDimTest : public NDArrayTests {
 public:
  sd::LongType dimensions[2] = {0, 2};
  sd::LongType inputShapeBuffer[10] = {3, 2, 3, 5, 15, 5, 1, 8192, 1, 99};
  float inputData[30] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                         16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0};
  float dataAssertion[30] = {1.0,  2.0,  3.0,  4.0, 5.0, 6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                             16.0, 17.0, 18.0, 0.0, 0.0, 21.0, 22.0, 23.0, 0.0, 0.0,  26.0, 27.0, 28.0, 0.0,  0.0};
  float result[30] = {0.0};
  float broadcastData[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0};
  sd::LongType broadcastShapeInfo[8] = {2, 2, 5, 5, 1, 8192, 1, 99};
  int opNum = 2;
  int dimensionLength = 2;
};

#ifndef SD_CUDA

TEST_F(BroadcastMultiDimTest, MultimDimTest) {
  auto tad = new shape::TAD();
  tad->init(inputShapeBuffer, dimensions, dimensionLength);
  tad->createTadOnlyShapeInfo();
  tad->createOffsets();
  functions::broadcast::Broadcast<float, float, float>::exec(opNum,
                                                             inputData,              // x
                                                             inputShapeBuffer,       // xShapeInfo
                                                             broadcastData,          // y
                                                             broadcastShapeInfo,     // yShapeInfo
                                                             result,                 // result
                                                             inputShapeBuffer,       // resultShapeInfo
                                                             dimensions,             // dimension
                                                             dimensionLength,        // dimensionLength
                                                             tad->tadOnlyShapeInfo,  // tadShapeInfo
                                                             tad->tadOffsets,        // tadOffset
                                                             tad->tadOnlyShapeInfo,  // tadShapeInfoZ
                                                             tad->tadOffsets, sd::LoopKind::COMMON, 0,
                                                             tad->numTads);  // tadOffsetZ

  for (int i = 0; i < 30; i++) {
    ASSERT_EQ(dataAssertion[i], result[i]);
  }

  delete tad;
}

#endif
