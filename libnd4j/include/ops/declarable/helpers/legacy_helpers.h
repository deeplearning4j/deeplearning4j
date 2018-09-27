/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author sgazeos@gmail.com
//
#ifndef __H_LEGACY_HELPERS__
#define __H_LEGACY_HELPERS__
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {
/*
    FORCEINLINE void reluDerivative(NDArray* theFirst, NDArray const* theSecond);
    FORCEINLINE void reluDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void relu6Derivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void leakyReluDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void eluDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void seluDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void cubeDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void reduceNorm1(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void sxeLossWithLogits(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void tanhDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void hardTanhDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void rationalTanhDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void rectifiedTanhDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void softSignDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void softPlusDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void sigmoidDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    FORCEINLINE void hardSigmoidDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
*/
    void reluDerivative(NDArray* theFirst, NDArray* theSecond);
    void reluDerivative(NDArray* theFirst, NDArray* theSecond, NDArray* theOutput);
    void relu6Derivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void leakyReluDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void eluDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void seluDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void cubeDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void reduceNorm1(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void sxeLossWithLogits(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void tanhDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void hardTanhDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void rationalTanhDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void rectifiedTanhDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void softSignDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void softPlusDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void sigmoidDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);
    void hardSigmoidDerivative(NDArray const* theFirst, NDArray const* theSecond, NDArray* theOutput);

}
}
}
#endif
