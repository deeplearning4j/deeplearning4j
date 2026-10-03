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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Ops whose helpers once computed with host lambdas. On CUDA the host result of weighted_cross_entropy_with_logits was
 * then dropped for the device copy, and per-class weights were ignored there; on the CPU per-class weights overwrote
 * the weights and targets inputs and squared the targets. thresholdedrelu and apply_sgd ran on the host on CUDA, where
 * a CUDA graph capturing them never sees the work.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class ClosedFormHelperOpsTest extends BaseNd4jTestWithBackends {

    private static double softplus(double t) {
        return Math.max(t, 0) + Math.log1p(Math.exp(-Math.abs(t)));
    }

    private static void assertClose(INDArray expected, INDArray actual) {
        assertEquals(expected.length(), actual.length());
        double[] e = expected.dup().data().asDouble();
        double[] a = actual.dup().data().asDouble();
        for (int i = 0; i < e.length; i++) {
            assertEquals(e[i], a[i], 1e-12 * Math.max(1, Math.abs(e[i])), "element " + i);
        }
    }

    /** loss = (1 - z) x + (1 + (w - 1) z) softplus(-x), with w per class (the last axis) or one for all. */
    private static INDArray expectedLoss(INDArray logits, INDArray targets, double[] classWeights) {
        long rows = logits.size(0), classes = logits.size(1);
        double[] out = new double[(int) (rows * classes)];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < classes; c++) {
                double x = logits.getDouble(r, c), z = targets.getDouble(r, c);
                double w = classWeights[classWeights.length == 1 ? 0 : c];
                out[(int) (r * classes + c)] = (1 - z) * x + (1 + (w - 1) * z) * softplus(-x);
            }
        }
        return Nd4j.createFromArray(out).reshape(rows, classes);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void weightedCrossEntropyWithLogits(Nd4jBackend backend) {
        INDArray logits = Nd4j.createFromArray(new double[][]{{-30, -1, 0}, {0.5, 2, 40}});
        INDArray targets = Nd4j.createFromArray(new double[][]{{0, 1, 0.5}, {1, 0, 0.25}});
        for (double[] classWeights : new double[][]{{3.0}, {0.5, 2.0, 4.0}}) {
            INDArray weights = classWeights.length == 1 ? Nd4j.scalar(classWeights[0]) : Nd4j.createFromArray(classWeights);
            INDArray logitsBefore = logits.dup(), targetsBefore = targets.dup(), weightsBefore = weights.dup();
            INDArray out = Nd4j.createUninitialized(logits.dataType(), logits.shape());

            Nd4j.exec(DynamicCustomOp.builder("weighted_cross_entropy_with_logits")
                    .addInputs(targets, logits, weights).addOutputs(out).build());

            assertClose(expectedLoss(logits, targets, classWeights), out);
            assertEquals(logitsBefore, logits);
            assertEquals(targetsBefore, targets);
            assertEquals(weightsBefore, weights);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void thresholdedReluKeepsOnlyValuesAboveTheThreshold(Nd4jBackend backend) {
        INDArray in = Nd4j.createFromArray(Double.NEGATIVE_INFINITY, -1.0, 0.5, 0.75, 1.0, 2.0, Double.NaN);
        INDArray out = Nd4j.createUninitialized(in.dataType(), in.shape());

        Nd4j.exec(DynamicCustomOp.builder("thresholdedrelu").addInputs(in).addOutputs(out)
                .addFloatingPointArguments(0.75).build());

        assertEquals(Nd4j.createFromArray(0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0), out);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void applySgdSubtractsTheScaledGradient(Nd4jBackend backend) {
        INDArray gradients = Nd4j.createFromArray(0.5, -1.0, 2.0);
        INDArray expected = Nd4j.createFromArray(1 - 0.5 * 0.1, 2 + 1.0 * 0.1, 3 - 2.0 * 0.1);

        INDArray parameters = Nd4j.createFromArray(1.0, 2.0, 3.0);
        INDArray out = Nd4j.createUninitialized(parameters.dataType(), parameters.shape());
        Nd4j.exec(DynamicCustomOp.builder("apply_sgd").addInputs(parameters, gradients).addOutputs(out)
                .addFloatingPointArguments(0.1).build());
        assertClose(expected, out);
        assertEquals(Nd4j.createFromArray(1.0, 2.0, 3.0), parameters);

        // In place: the output is the parameters array
        Nd4j.exec(DynamicCustomOp.builder("apply_sgd").addInputs(parameters, gradients).addOutputs(parameters)
                .addFloatingPointArguments(0.1).build());
        assertClose(expected, parameters);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
