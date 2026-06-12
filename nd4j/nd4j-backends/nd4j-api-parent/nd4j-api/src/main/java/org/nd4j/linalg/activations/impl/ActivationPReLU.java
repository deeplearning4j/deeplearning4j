/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

@EqualsAndHashCode(callSuper = false)
@Getter
public class ActivationPReLU extends BaseActivationFunction {

    private INDArray alpha;
    private long[] sharedAxes = null;

    public ActivationPReLU(INDArray alpha, long[] sharedAxes) {
        this.alpha = alpha;
        this.sharedAxes = sharedAxes;
    }

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        // If alpha is a view (e.g. a slice of a larger parameter array), dup it to ensure
        // the C++ native op receives a contiguous array with offset=0. Without this dup,
        // the native code may read alpha at the wrong buffer offset and see 0 instead of
        // the actual parameter value, producing f(x) = 0*x = 0 for all negative inputs.
        INDArray alphaForOp = alpha.isView() ? alpha.dup() : alpha;
        DynamicCustomOp.DynamicCustomOpsBuilder prelu = DynamicCustomOp.builder("prelu")
                .addOutputs(in).addInputs(in, alphaForOp);
        if (sharedAxes != null) {
            for (long axis: sharedAxes) {
                prelu.addIntegerArguments(axis);
            }
        }
        Nd4j.getExecutioner().execAndReturn(prelu.build());
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);
        INDArray dLdalpha = alpha.ulike();
        INDArray outTemp = in.ulike();
        // If alpha is a view (e.g. a slice of a larger parameter array), dup it to ensure
        // the C++ native op receives a contiguous array with offset=0. Without this dup,
        // the native code may read alpha at the wrong buffer offset and see 0 instead of
        // the actual parameter value, producing dLdI = grO * 0 = 0 for all negative inputs.
        INDArray alphaForOp = alpha.isView() ? alpha.dup() : alpha;
        DynamicCustomOp.DynamicCustomOpsBuilder preluBp = DynamicCustomOp.builder("prelu_bp")
                .addInputs(in, alphaForOp, epsilon)
                .addOutputs(outTemp, dLdalpha);

        if (sharedAxes != null) {
            for (long axis: sharedAxes) {
                preluBp.addIntegerArguments(axis);
            }
        }
        Nd4j.exec(preluBp.build());
        in.assign(outTemp);
        return new Pair<>(in, dLdalpha);
    }

    @Override
    public String toString() {
        return "prelu";
    }
}