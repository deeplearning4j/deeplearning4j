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

package org.eclipse.deeplearning4j.nd4j.linalg.topk;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.GradCheckUtil;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Gradient check for the TopK op.
 * TopK(x, k) -> (values, indices) where TopK operates on the last axis.
 * Only the values output has a real gradient; the indices are integers and have no gradient.
 * doDiff scatters gradValues back to the original positions using a one-hot matrix.
 */
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class TopKGradientTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * 1-D input, k=3 out of n=6.
     * x has values well-separated so the top-k set is stable under eps=1e-5 perturbations.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTopKGradient1D(Nd4jBackend backend) {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        // x = [10, 1, 8, 2, 9, 3]  ->  top-3 sorted = [10, 9, 8] at positions 0, 4, 2
        // Gaps: 8-3 = 5, well above eps
        INDArray xArr = Nd4j.createFromArray(new double[]{10.0, 1.0, 8.0, 2.0, 9.0, 3.0});

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", xArr);

        SDVariable[] topkOut = sd.nn().topK(x, 3, true);
        SDVariable topkValues = topkOut[0];  // shape [3]

        // loss = sum(topkValues^2) — scalar
        SDVariable loss = sd.sum(topkValues.mul(topkValues));
        loss.markAsLoss();

        boolean ok = GradCheckUtil.checkGradients(sd, null);
        assertTrue(ok, "TopK gradient check failed for 1D input");
    }

    /**
     * 2-D input [3, 6], k=3.
     * Values chosen so the top-3 boundary per row is far from adjacent values.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTopKGradient2D(Nd4jBackend backend) {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        // Row 0: top-3 = [10,9,8], 4th = 3  (gap 5)
        // Row 1: top-3 = [9,7,6],  4th = 5  (gap 1)
        // Row 2: top-3 = [8,7,4],  4th = 3  (gap 1)
        INDArray xArr = Nd4j.create(new double[][]{
                {10.0, 1.0, 8.0, 2.0, 9.0, 3.0},
                { 5.0, 7.0, 1.0, 9.0, 2.0, 6.0},
                { 3.0, 8.0, 4.0, 1.0, 7.0, 2.0}
        }).castTo(DataType.DOUBLE);

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", xArr);

        SDVariable[] topkOut = sd.nn().topK(x, 3, true);
        SDVariable topkValues = topkOut[0];  // shape [3, 3]

        // loss = sum(topkValues^2) — scalar
        SDVariable loss = sd.sum(topkValues.mul(topkValues));
        loss.markAsLoss();

        boolean ok = GradCheckUtil.checkGradients(sd, null);
        assertTrue(ok, "TopK gradient check failed for 2D input");
    }

    /**
     * 2-D input [2, 8], k=2 — tests a different k/n ratio.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTopKGradient2D_k2(Nd4jBackend backend) {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);

        // Row 0: top-2 = [20, 15], 3rd = 8  (gap 7)
        // Row 1: top-2 = [18, 12], 3rd = 7  (gap 5)
        INDArray xArr = Nd4j.create(new double[][]{
                { 3.0, 20.0,  5.0, 15.0,  1.0,  8.0,  2.0,  4.0},
                { 7.0,  2.0, 18.0,  1.0, 12.0,  3.0,  6.0,  4.0}
        }).castTo(DataType.DOUBLE);

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", xArr);

        SDVariable[] topkOut = sd.nn().topK(x, 2, true);
        SDVariable topkValues = topkOut[0];  // shape [2, 2]

        SDVariable loss = sd.sum(topkValues.mul(topkValues));
        loss.markAsLoss();

        // Diagnostic: check row 1 of xArr directly via the SameDiff graph (slice to 1D row)
        {
            // Run top_k on each row separately to verify correctness
            INDArray row0 = xArr.getRow(0);
            INDArray row1 = xArr.getRow(1);
            INDArray[] tk0 = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.TopK(row0, 2, true));
            INDArray[] tk1 = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.TopK(row1, 2, true));
            System.out.println("DIAG row0_tk val=" + tk0[0] + " idx=" + tk0[1]);
            System.out.println("DIAG row1_tk val=" + tk1[0] + " idx=" + tk1[1]);
            System.out.println("DIAG row0 is: " + row0.isView() + " contiguous=" + row0.isRowVector());
            System.out.println("DIAG row1 is: " + row1.isView() + " contiguous=" + row1.isRowVector());
            // Try dup() of row1 to force contiguous copy
            INDArray row1dup = row1.dup('c');
            INDArray[] tk1dup = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.TopK(row1dup, 2, true));
            System.out.println("DIAG row1dup_tk val=" + tk1dup[0] + " idx=" + tk1dup[1]);
        }

        // Also test: full 2D with pre-zerod output
        {
            INDArray freshX = Nd4j.create(new double[][]{
                    { 3.0, 20.0, 5.0, 15.0, 1.0, 8.0, 2.0, 4.0 },
                    { 7.0, 2.0, 18.0, 1.0, 12.0, 3.0, 6.0, 4.0 }
            }).castTo(DataType.DOUBLE);
            INDArray[] tkF = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.TopK(freshX, 2, true));
            System.out.println("FRESH 2D full val[1,0]=" + tkF[0].getDouble(1,0) + " val[1,1]=" + tkF[0].getDouble(1,1));
            System.out.println("FRESH 2D full idx[1,0]=" + tkF[1].getLong(1,0) + " idx[1,1]=" + tkF[1].getLong(1,1));
        }

        boolean ok = GradCheckUtil.checkGradients(sd, null);
        assertTrue(ok, "TopK gradient check failed for 2D input k=2");
    }
}
