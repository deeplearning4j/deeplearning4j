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
// Three-operand random kernels. Split from random.cu so each random
// translation unit instantiates one kernel family at a time (see
// random_impl.h for the rationale).
//
#include <loops/cuda/random_impl.h>

namespace functions {
namespace random {

template <>
SD_HOST void RandomFunction<float>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                     sd::Pointer stateHost, void const* vx,
                                                     sd::LongType const* xShapeBuffer, void const* vy,
                                                     sd::LongType const* yShapeBuffer, void* vz,
                                                     sd::LongType const* zShapeBuffer, void* vextraArguments) {
 auto x = reinterpret_cast<float const*>(vx);
 auto y = reinterpret_cast<float const*>(vy);
 auto z = reinterpret_cast<float*>(vz);
 auto extraArguments = reinterpret_cast<float*>(vextraArguments);

 // this macro builds bunch of IF/ELSE selectors for kernel launch
 DISPATCH_SIMPLE(randomTriple, float,
                 PARAMS(stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                 OPS_A(RANDOM_OPS))

 sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<float16>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                       sd::Pointer stateHost, void const* vx,
                                                       sd::LongType const* xShapeBuffer, void const* vy,
                                                       sd::LongType const* yShapeBuffer, void* vz,
                                                       sd::LongType const* zShapeBuffer, void* vextraArguments) {
 auto x = reinterpret_cast<float16 const*>(vx);
 auto y = reinterpret_cast<float16 const*>(vy);
 auto z = reinterpret_cast<float16*>(vz);
 auto extraArguments = reinterpret_cast<float16*>(vextraArguments);

 // this macro builds bunch of IF/ELSE selectors for kernel launch
 DISPATCH_SIMPLE(randomTriple, float16,
                 PARAMS(stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                 OPS_A(RANDOM_OPS))

 sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<bfloat16>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                        sd::Pointer stateHost, void const* vx,
                                                        sd::LongType const* xShapeBuffer, void const* vy,
                                                        sd::LongType const* yShapeBuffer, void* vz,
                                                        sd::LongType const* zShapeBuffer, void* vextraArguments) {
 auto x = reinterpret_cast<bfloat16 const*>(vx);
 auto y = reinterpret_cast<bfloat16 const*>(vy);
 auto z = reinterpret_cast<bfloat16*>(vz);
 auto extraArguments = reinterpret_cast<bfloat16*>(vextraArguments);

 // this macro builds bunch of IF/ELSE selectors for kernel launch
 DISPATCH_SIMPLE(randomTriple, bfloat16,
                 PARAMS(stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                 OPS_A(RANDOM_OPS))

 sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

template <>
SD_HOST void RandomFunction<double>::executeCudaTriple(dim3& launchDims, cudaStream_t* stream, int opNum,
                                                      sd::Pointer stateHost, void const* vx,
                                                      sd::LongType const* xShapeBuffer, void const* vy,
                                                      sd::LongType const* yShapeBuffer, void* vz,
                                                      sd::LongType const* zShapeBuffer, void* vextraArguments) {
 auto x = reinterpret_cast<double const*>(vx);
 auto y = reinterpret_cast<double const*>(vy);
 auto z = reinterpret_cast<double*>(vz);
 auto extraArguments = reinterpret_cast<double*>(vextraArguments);

 // this macro builds bunch of IF/ELSE selectors for kernel launch
 DISPATCH_SIMPLE(randomTriple, double,
                 PARAMS(stateHost, x, xShapeBuffer, y, yShapeBuffer, z, zShapeBuffer, extraArguments),
                 OPS_A(RANDOM_OPS))

 sd::DebugHelper::checkErrorCode(stream, "RandomFunction executeCudaSingle(...) failed");
}

}  // namespace random
}  // namespace functions
