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
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * alpha_dropout_bp: the forward maps a kept element to alpha * x + alpha1 and a dropped one to alpha * beta + alpha1,
 * and its mask records the keep decisions (1 kept, 0 dropped), so the gradient is gradOut * mask * alpha. With a keep
 * probability of 1 every element is kept: the gradient passes through scaled by alpha (it used to be zeroed).
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class AlphaDropoutBackpropTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void gradientIsTheKeptGradOutScaledByAlpha(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(29);
        double alpha = 1.7, alpha1 = -0.3, beta = -1.75;
        INDArray input = Nd4j.rand(DataType.DOUBLE, 3, 4).subi(0.5);
        INDArray gradOut = Nd4j.rand(DataType.DOUBLE, 3, 4).subi(0.5);
        for (double keepProbability : new double[]{1.0, 0.5}) {
            INDArray mask = keepProbability == 1.0 ? Nd4j.ones(DataType.DOUBLE, 3, 4)
                    : Nd4j.createFromArray(new double[][]{{1, 0, 1, 1}, {0, 0, 1, 0}, {1, 1, 0, 1}});
            INDArray expected = gradOut.mul(mask).muli(alpha);
            INDArray[] out = Nd4j.exec(DynamicCustomOp.builder("alpha_dropout_bp").addInputs(input, mask, gradOut)
                    .addFloatingPointArguments(keepProbability, alpha, alpha1, beta).addIntegerArguments(1).build());
            double maxDiff = out[0].sub(expected).amaxNumber().doubleValue();
            assertTrue(maxDiff <= 1e-12, "alpha_dropout_bp, keep probability " + keepProbability + ": max |diff| "
                    + maxDiff + "\n" + out[0] + "\nvs\n" + expected);
        }
    }
}
