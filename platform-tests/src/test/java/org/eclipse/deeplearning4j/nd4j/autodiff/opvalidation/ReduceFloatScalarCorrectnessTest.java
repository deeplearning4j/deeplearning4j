/* ******************************************************************************
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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.reduce.floating.Norm2;
import org.nd4j.linalg.api.ops.impl.reduce.floating.SquaredNorm;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Validates that float reduce-to-scalar ops return mathematically correct values.
 *
 * SquaredNorm = sum(x^2), Norm2 = sqrt(sum(x^2)). Both dispatch through the same
 * reduce_float scalar kernel (execReduceFloatScalar) and differ only in their
 * postProcess step (identity vs sqrt), so any divergence between the two isolates a
 * backend reduction bug rather than an op-definition bug. This pins down a reported
 * CUDA regression where reduceNumber(SquaredNorm) returned half of sum(x^2).
 */
public class ReduceFloatScalarCorrectnessTest {

    @Test
    public void testFloatReduceScalarValues() {
        INDArray x = Nd4j.createFromArray(new float[]{1f, 2f, 3f, 4f});
        double expSumSq = 1 + 4 + 9 + 16; // 30

        // Op path: execReduceFloatScalar with opNum 7 (SquaredNorm) and opNum 3 (Norm2).
        double squaredNormOp = Nd4j.getExecutioner().exec(new SquaredNorm(x)).getDouble(0);
        double norm2Op = Nd4j.getExecutioner().exec(new Norm2(x)).getDouble(0);

        // Convenience accessors (may route through a different reduction path).
        double norm2Number = x.norm2Number().doubleValue();
        double meanNumber = x.meanNumber().doubleValue();

        assertEquals(expSumSq, squaredNormOp, 1e-4, "SquaredNorm op should equal sum(x^2)=30");
        assertEquals(Math.sqrt(expSumSq), norm2Op, 1e-4, "Norm2 op should equal sqrt(sum(x^2))");
        assertEquals(Math.sqrt(expSumSq), norm2Number, 1e-4, "norm2Number should equal sqrt(sum(x^2))");
        assertEquals(2.5, meanNumber, 1e-4, "meanNumber should equal sum(x)/n");
    }

    /**
     * Same checks but on DOUBLE input. testMatrixBandPart's l2Loss runs on a DOUBLE matrix and its
     * gradient check showed the forward loss = 0.25*sum(x^2) (half of 0.5*sum(x^2)), i.e.
     * reduceNumber(SquaredNorm) returned half on DOUBLE while the FLOAT path above is correct.
     * This isolates whether the bug is specific to the DOUBLE template instantiation of reduce_float.
     */
    @Test
    public void testDoubleReduceScalarValues() {
        INDArray x = Nd4j.createFromArray(new double[]{1.0, 2.0, 3.0, 4.0});
        double expSumSq = 1 + 4 + 9 + 16; // 30

        double squaredNormOp = Nd4j.getExecutioner().exec(new SquaredNorm(x)).getDouble(0);
        double norm2Op = Nd4j.getExecutioner().exec(new Norm2(x)).getDouble(0);

        assertEquals(expSumSq, squaredNormOp, 1e-6, "SquaredNorm(DOUBLE) should equal sum(x^2)=30");
        assertEquals(Math.sqrt(expSumSq), norm2Op, 1e-6, "Norm2(DOUBLE) should equal sqrt(sum(x^2))");
    }

    /**
     * l2_loss = 1/2 * sum(x^2). Guards against the regression where the standalone op and the
     * SameDiff graph evaluation of L2Loss diverged (one returned half), using the exact
     * testMatrixBandPart input.
     */
    @Test
    public void testL2LossSameDiffVsOp() {
        INDArray input = Nd4j.createFromArray(new double[]{
                0.7788, 0.8012, 0.7244, 0.2309,
                0.7271, 0.1804, 0.5056, 0.8925,
                0.5461, 0.9234, 0.0856, 0.7938}).reshape(3, 4);
        double sumSq = input.mul(input).sumNumber().doubleValue();
        double expected = 0.5 * sumSq; // l2_loss = 1/2 * sum(x^2)

        // 1. Standalone custom op (l2_loss.cpp directly)
        INDArray opOut = Nd4j.exec(DynamicCustomOp.builder("l2_loss").addInputs(input).build())[0];
        double opVal = opOut.getDouble(0);

        // 2. Same op evaluated inside a SameDiff graph
        SameDiff sd = SameDiff.create();
        SDVariable v = sd.var("in", input);
        SDVariable loss = sd.loss.l2Loss(v);
        double sdVal = loss.eval().getDouble(0);

        assertEquals(expected, opVal, 1e-6, "standalone l2_loss op should equal 0.5*sum(x^2)");
        assertEquals(expected, sdVal, 1e-6, "l2Loss SameDiff eval should equal 0.5*sum(x^2)");
    }

    /**
     * Guards that SquaredNorm is correct for rank>1 (2D) inputs as well as flattened 1D — the
     * l2_loss regression returned half on a [3,4] DOUBLE input (xLength=12) while a 1D length-4
     * input was correct, so this checks length- and shape-independence.
     */
    @Test
    public void testSquaredNorm2DvsFlat() {
        INDArray input2D = Nd4j.createFromArray(new double[]{
                0.7788, 0.8012, 0.7244, 0.2309,
                0.7271, 0.1804, 0.5056, 0.8925,
                0.5461, 0.9234, 0.0856, 0.7938}).reshape(3, 4);
        double manualSumSq = input2D.mul(input2D).sumNumber().doubleValue();

        double sn2D = Nd4j.getExecutioner().exec(new SquaredNorm(input2D)).getDouble(0);
        INDArray flat12 = input2D.dup().reshape(12);
        double sn1D12 = Nd4j.getExecutioner().exec(new SquaredNorm(flat12)).getDouble(0);

        // Length-12 1D of simple values too: 1..12, sum sq = 650
        INDArray simple12 = Nd4j.createFromArray(new double[]{1,2,3,4,5,6,7,8,9,10,11,12});
        double snSimple12 = Nd4j.getExecutioner().exec(new SquaredNorm(simple12)).getDouble(0);

        assertEquals(manualSumSq, sn2D, 1e-6, "SquaredNorm on 2D [3,4] should equal sum(x^2)");
        assertEquals(manualSumSq, sn1D12, 1e-6, "SquaredNorm on flat [12] should equal sum(x^2)");
        assertEquals(650.0, snSimple12, 1e-6, "SquaredNorm on 1..12 should equal 650");
    }

    /**
     * Regression guard for the in-place scalar double-apply bug: output->applyScalar(Divide, 2.0)
     * divided by 4 instead of 2 on CUDA (the l2_loss path). Covers in-place vs fresh-target,
     * scalar-length vs vector, and Divide vs Multiply.
     */
    @Test
    public void testScalarDivideInPlace() {
        INDArray s1 = Nd4j.scalar(5.2282658);
        s1.divi(2.0);                                   // in-place scalar divide (the l2_loss path)

        INDArray s2 = Nd4j.scalar(5.2282658);
        INDArray s2r = s2.div(2.0);                     // fresh-target scalar divide

        INDArray v = Nd4j.createFromArray(new double[]{5.2282658, 10.0});
        v.divi(2.0);                                    // in-place vector divide

        INDArray m = Nd4j.scalar(5.2282658);
        m.muli(0.5);                                    // in-place scalar multiply

        assertEquals(2.6141329, s1.getDouble(0), 1e-6, "in-place scalar divi(2.0) should divide by 2");
        assertEquals(2.6141329, s2r.getDouble(0), 1e-6, "fresh scalar div(2.0) should divide by 2");
        assertEquals(2.6141329, v.getDouble(0), 1e-6, "in-place vector divi(2.0) should divide by 2");
        assertEquals(2.6141329, m.getDouble(0), 1e-6, "in-place scalar muli(0.5) should multiply by 0.5");
    }
}
