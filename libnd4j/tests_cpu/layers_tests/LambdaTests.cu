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
#include <array/ExtraArguments.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>

#include "testlayers.h"

using namespace sd;

class LambdaTests : public testing::Test {
 public:
  LambdaTests() {
    printf("\n");
    fflush(stdout);
  }
};

template <typename Lambda>
SD_KERNEL void runLambda(double *input, double *output, sd::LongType length, Lambda lambda) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (sd::LongType e = tid; e < length; e += gridDim.x * blockDim.x) {
    output[e] = lambda(input[e]);
  }
}

void launcher(cudaStream_t *stream, double *input, double *output, sd::LongType length) {
  // auto f = [] SD_HOST_DEVICE (double x) -> double {
  //        return x + 1.;
  //};
  auto f = LAMBDA_D(x) { return x + 1.; };

  runLambda<<<128, 128, 128, *stream>>>(input, output, length, f);
}
