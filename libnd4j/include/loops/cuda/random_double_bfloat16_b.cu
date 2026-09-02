/*
 * ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership. Unless required by applicable
 *  law or agreed to in writing, software distributed under the License is
 *  distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied. See the License for the specific
 *  language governing permissions and limitations under the License.
 *
 *  SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//
// Per-type, per-op-half random kernels. CUDA 12.6/12.9 nvcc segfaults
// (cicc) and Windows ptxas access-violates when every random op for a
// family is expanded in one translation unit, so each (family, dtype,
// op-half) pair gets its own small TU.
//
#include <loops/cuda/random_impl.h>

namespace functions {
namespace random {
#define RANDOM_OPS_HALF (9, BinomialDistributionEx), (10, LogNormalDistribution), (11, TruncatedNormalDistribution), (12, AlphaDropOut), (13, ExponentialDistribution), (14, ExponentialDistributionInv), (15, PoissonDistribution), (16, GammaDistribution)

DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, bfloat16,
                      INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void* z, sd::LongType const* zShapeBuffer, void* extraArguments),
                      PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS_HALF))

template <>
SD_HOST void RandomFunction<bfloat16>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                      sd::Pointer stateHost, void const* vx, sd::LongType const* xShapeBuffer, void* vz, sd::LongType const* zShapeBuffer, void* vextraArguments) {
 auto x = reinterpret_cast<bfloat16 const*>(vx);
 auto z = reinterpret_cast<bfloat16*>(vz);
 auto extraArguments = reinterpret_cast<bfloat16*>(vextraArguments);

 // this macro builds bunch of IF/ELSE selectors for kernel launch
 DISPATCH_SIMPLE(randomDouble, float16, PARAMS(stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS_HALF))

 sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

}  // namespace random
}  // namespace functions
