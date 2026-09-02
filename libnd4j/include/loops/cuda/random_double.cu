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
// Two-operand random kernels. Split from random.cu so each random
// translation unit instantiates one kernel family at a time (see
// random_impl.h for the rationale).
//
#include <loops/cuda/random_impl.h>

namespace functions {
namespace random {

// here we generate kernels for target operations
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, float,
                      INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void* z,
                            sd::LongType const* zShapeBuffer, void* extraArguments),
                      PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, double,
                      INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void* z,
                            sd::LongType const* zShapeBuffer, void* extraArguments),
                      PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, float16,
                      INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void* z,
                            sd::LongType const* zShapeBuffer, void* extraArguments),
                      PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))
DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, bfloat16,
                      INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void* z,
                            sd::LongType const* zShapeBuffer, void* extraArguments),
                      PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A(RANDOM_OPS))

template <>
SD_HOST void RandomFunction<float>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                     sd::Pointer stateHost, void const* vx,
                                                     sd::LongType const* xShapeBuffer, void* vz,
                                                     sd::LongType const* zShapeBuffer, void* vextraArguments) {
 auto x = reinterpret_cast<float const*>(vx);
 auto z = reinterpret_cast<float*>(vz);
 auto extraArguments = reinterpret_cast<float*>(vextraArguments);

 // this macro builds bunch of IF/ELSE selectors for kernel launch
 DISPATCH_SIMPLE(randomDouble, float, PARAMS(stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments),
                 OPS_A(RANDOM_OPS))

 sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<float16>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                       sd::Pointer stateHost, void const* vx,
                                                       sd::LongType const* xShapeBuffer, void* vz,
                                                       sd::LongType const* zShapeBuffer, void* vextraArguments) {
 auto x = reinterpret_cast<float16 const*>(vx);
 auto z = reinterpret_cast<float16*>(vz);
 auto extraArguments = reinterpret_cast<float16*>(vextraArguments);

 // this macro builds bunch of IF/ELSE selectors for kernel launch
 DISPATCH_SIMPLE(randomDouble, float16, PARAMS(stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments),
                 OPS_A(RANDOM_OPS))

 sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<bfloat16>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                        sd::Pointer stateHost, void const* vx,
                                                        sd::LongType const* xShapeBuffer, void* vz,
                                                        sd::LongType const* zShapeBuffer, void* vextraArguments) {
 auto x = reinterpret_cast<bfloat16 const*>(vx);
 auto z = reinterpret_cast<bfloat16*>(vz);
 auto extraArguments = reinterpret_cast<bfloat16*>(vextraArguments);

 // this macro builds bunch of IF/ELSE selectors for kernel launch
 DISPATCH_SIMPLE(randomDouble, bfloat16, PARAMS(stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments),
                 OPS_A(RANDOM_OPS))

 sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<double>::executeCudaDouble(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                      sd::Pointer stateHost, void const* vx,
                                                      sd::LongType const* xShapeBuffer, void* vz,
                                                      sd::LongType const* zShapeBuffer, void* vextraArguments) {
 auto x = reinterpret_cast<double const*>(vx);
 auto z = reinterpret_cast<double*>(vz);
 auto extraArguments = reinterpret_cast<double*>(vextraArguments);

 // this macro builds bunch of IF/ELSE selectors for kernel launch
 DISPATCH_SIMPLE(randomDouble, double, PARAMS(stateHost, x, xShapeBuffer, z, zShapeBuffer, extraArguments),
                 OPS_A(RANDOM_OPS))

 sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

}  // namespace random
}  // namespace functions
