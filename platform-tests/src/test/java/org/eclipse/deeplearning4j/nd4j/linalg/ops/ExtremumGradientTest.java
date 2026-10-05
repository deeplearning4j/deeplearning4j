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
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * maximum_bp and minimum_bp give dL/dz to the input z takes its value from, half to each where x == y, and sum each
 * gradient over the axes its input was broadcast along. On CUDA minimum_bp once computed on the host and then marked
 * the device copy current, so every gradient came back 0; on CPU a scalar y took all of dL/dz or none of it by
 * comparing the two array pointers, and a broadcast y compared against x's gradient instead of x.
 */
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@NativeTag
public class ExtremumGradientTest extends BaseNd4jTestWithBackends {

    private static INDArray[] backprop(String op, INDArray x, INDArray y, INDArray eps) {
        INDArray gradX = Nd4j.createUninitialized(x.dataType(), x.shape());
        INDArray gradY = Nd4j.createUninitialized(y.dataType(), y.shape());
        Nd4j.exec(DynamicCustomOp.builder(op).addInputs(x, y, eps).addOutputs(gradX, gradY).build());
        return new INDArray[]{gradX, gradY};
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void sameShapeSplitsTies(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(1.0, 5.0, 3.0, 2.0);
        INDArray y = Nd4j.createFromArray(4.0, 5.0, 1.0, 2.0);
        INDArray eps = Nd4j.createFromArray(1.0, 2.0, 3.0, 4.0);

        INDArray[] max = backprop("maximum_bp", x, y, eps);
        assertEquals(Nd4j.createFromArray(0.0, 1.0, 3.0, 2.0), max[0]);
        assertEquals(Nd4j.createFromArray(1.0, 1.0, 0.0, 2.0), max[1]);

        INDArray[] min = backprop("minimum_bp", x, y, eps);
        assertEquals(Nd4j.createFromArray(1.0, 1.0, 0.0, 2.0), min[0]);
        assertEquals(Nd4j.createFromArray(0.0, 1.0, 3.0, 2.0), min[1]);
    }

    /** y [3] is broadcast along x's rows, so its gradient sums its share of every row. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void broadcastRowSumsItsShares(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(new double[][]{{1, 6, 3}, {4, 2, 9}});
        INDArray y = Nd4j.createFromArray(2.0, 2.0, 5.0);
        INDArray eps = Nd4j.createFromArray(new double[][]{{1, 2, 3}, {4, 5, 6}});

        INDArray[] max = backprop("maximum_bp", x, y, eps);
        assertEquals(Nd4j.createFromArray(new double[][]{{0, 2, 0}, {4, 2.5, 6}}), max[0]);
        assertEquals(Nd4j.createFromArray(1.0, 2.5, 3.0), max[1]);

        INDArray[] min = backprop("minimum_bp", x, y, eps);
        assertEquals(Nd4j.createFromArray(new double[][]{{1, 0, 3}, {0, 2.5, 0}}), min[0]);
        assertEquals(Nd4j.createFromArray(4.0, 4.5, 6.0), min[1]);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void scalarTakesItsShareOfEveryElement(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(1.0, 3.0, 2.0);
        INDArray y = Nd4j.scalar(2.0);
        INDArray eps = Nd4j.createFromArray(1.0, 2.0, 3.0);

        INDArray[] max = backprop("maximum_bp", x, y, eps);
        assertEquals(Nd4j.createFromArray(0.0, 2.0, 1.5), max[0]);
        assertEquals(Nd4j.scalar(2.5), max[1]);

        INDArray[] min = backprop("minimum_bp", x, y, eps);
        assertEquals(Nd4j.createFromArray(1.0, 0.0, 1.5), min[0]);
        assertEquals(Nd4j.scalar(3.5), min[1]);

        INDArray[] swappedMax = backprop("maximum_bp", y, x, eps);
        assertEquals(max[1], swappedMax[0], "rank-zero first destination");
        assertEquals(max[0], swappedMax[1], "vector second destination");
        INDArray[] swappedMin = backprop("minimum_bp", y, x, eps);
        assertEquals(min[1], swappedMin[0], "rank-zero first destination");
        assertEquals(min[0], swappedMin[1], "vector second destination");
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void exceptionalEpsilonStillMultipliesZeroShares(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(1.0, 2.0, -0.0, 1.0);
        INDArray y = Nd4j.createFromArray(2.0, 1.0, 0.0, 2.0);
        INDArray eps = Nd4j.createFromArray(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 2.0, -1.0);
        for (String name : new String[]{"maximum_bp", "minimum_bp"}) {
            INDArray[] gradients = backprop(name, x, y, eps);
            int losing = name.equals("maximum_bp") ? 0 : 1;
            assertEquals(Double.NaN, gradients[0].getDouble(losing), name);
            assertEquals(Double.POSITIVE_INFINITY, gradients[0].getDouble(1 - losing), name);
            assertEquals(Double.POSITIVE_INFINITY, gradients[1].getDouble(losing), name);
            assertEquals(Double.NaN, gradients[1].getDouble(1 - losing), name);
            assertEquals(1.0, gradients[0].getDouble(2), name + " signed-zero tie");
            assertEquals(1.0, gradients[1].getDouble(2), name + " signed-zero tie");
            assertEquals(Double.doubleToRawLongBits(-0.0),
                    Double.doubleToRawLongBits(gradients[name.equals("maximum_bp") ? 0 : 1].getDouble(3)),
                    name + " negative epsilon times zero");
        }
    }

    @ParameterizedTest
    @MethodSource("configs")
    public void bothOperandsBroadcastIntoOffsetOutputs(Nd4jBackend backend) {
        INDArray x = Nd4j.createFromArray(new double[][]{{2}, {5}}).dup('f');
        INDArray y = Nd4j.createFromArray(new double[][]{{2, 4, 5}});
        INDArray eps = Nd4j.createFromArray(new double[][]{{1, 2, 3}, {4, 5, 6}}).dup('f');
        for (String name : new String[]{"maximum_bp", "minimum_bp"}) {
            INDArray xParent = Nd4j.valueArrayOf(new long[]{2, 2}, -99.0, x.dataType());
            INDArray yParent = Nd4j.valueArrayOf(new long[]{2, 3}, -99.0, y.dataType());
            INDArray gradX = xParent.getColumn(1).reshape(2, 1);
            INDArray gradY = yParent.getRow(1).reshape(1, 3);
            assertEquals(2, gradX.stride()[0], "gradient destination remains stepped");
            Nd4j.exec(DynamicCustomOp.builder(name).addInputs(x, y, eps).addOutputs(gradX, gradY).build());
            INDArray expectedX = Nd4j.createFromArray(new double[][]{
                    {name.equals("maximum_bp") ? 0.5 : 5.5}, {name.equals("maximum_bp") ? 12 : 3}});
            INDArray expectedY = Nd4j.createFromArray(new double[][]{name.equals("maximum_bp")
                    ? new double[]{0.5, 2, 6} : new double[]{4.5, 5, 3}});
            assertEquals(expectedX, gradX, name);
            assertEquals(expectedY, gradY, name);
            assertEquals(expectedX.getDouble(0), xParent.getDouble(0, 1), name);
            assertEquals(expectedX.getDouble(1), xParent.getDouble(1, 1), name);
            assertEquals(Nd4j.valueArrayOf(new long[]{2}, -99.0, x.dataType()), xParent.getColumn(0), name);
            assertEquals(Nd4j.valueArrayOf(new long[]{3}, -99.0, y.dataType()), yParent.getRow(0), name);
        }
    }

    /** max(x, x) and min(x, x) are x: the two halves of every tie add up to the whole gradient. */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void extremumOfAVariableWithItselfHasGradientOne(Nd4jBackend backend) {
        INDArray weights = Nd4j.createFromArray(1.0, 2.0, 3.0);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.var("in", Nd4j.createFromArray(1.0, -2.0, 3.0));
        SDVariable w = sd.constant("w", weights);
        sd.max(in, in).mul(w).sum().add(sd.min(in, in).mul(w).sum()).markAsLoss();

        assertEquals(weights.mul(2), sd.calculateGradients(Collections.emptyMap(), "in").get("in"));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
