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

package org.eclipse.deeplearning4j.nd4j.linalg.custom;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@NativeTag
public class VisionTransformerOpsTests extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    // ==================== EMA Update ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmaUpdateBasic(Nd4jBackend backend) {
        // output = decay * shadow + (1 - decay) * model
        INDArray model = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0});
        INDArray shadow = Nd4j.create(new double[]{0.0, 0.0, 0.0, 0.0});
        double decay = 0.9;

        INDArray output = Nd4j.create(DataType.DOUBLE, 4);

        DynamicCustomOp op = DynamicCustomOp.builder("ema_update")
                .addInputs(model, shadow)
                .addOutputs(output)
                .addFloatingPointArguments(decay)
                .build();
        Nd4j.getExecutioner().exec(op);

        // expected: 0.9 * 0 + 0.1 * model = 0.1 * model
        INDArray expected = model.mul(1.0 - decay);
        assertEquals(expected, output);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmaUpdateConvergence(Nd4jBackend backend) {
        // After many updates, shadow should approach model
        INDArray model = Nd4j.create(new double[]{10.0, 20.0, 30.0});
        INDArray shadow = Nd4j.zeros(DataType.DOUBLE, 3);
        double decay = 0.9;

        INDArray output = Nd4j.create(DataType.DOUBLE, 3);
        for (int i = 0; i < 100; i++) {
            DynamicCustomOp op = DynamicCustomOp.builder("ema_update")
                    .addInputs(model, shadow)
                    .addOutputs(output)
                    .addFloatingPointArguments(decay)
                    .build();
            Nd4j.getExecutioner().exec(op);
            shadow.assign(output);
        }

        // After 100 iterations with decay=0.9, shadow ≈ model
        for (int i = 0; i < 3; i++) {
            assertEquals(model.getDouble(i), output.getDouble(i), 1e-3,
                    "EMA should converge to model after many steps");
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmaUpdateDecayOne(Nd4jBackend backend) {
        // decay=1.0 means output = shadow (no update)
        INDArray model = Nd4j.create(new double[]{1.0, 2.0, 3.0});
        INDArray shadow = Nd4j.create(new double[]{4.0, 5.0, 6.0});
        INDArray output = Nd4j.create(DataType.DOUBLE, 3);

        DynamicCustomOp op = DynamicCustomOp.builder("ema_update")
                .addInputs(model, shadow)
                .addOutputs(output)
                .addFloatingPointArguments(1.0)
                .build();
        Nd4j.getExecutioner().exec(op);

        assertEquals(shadow, output);
    }

    // ==================== EMA Update Backward ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmaUpdateBp(Nd4jBackend backend) {
        // dModel = (1-decay) * gradOut, dShadow = decay * gradOut
        INDArray model = Nd4j.randn(DataType.DOUBLE, 4);
        INDArray shadow = Nd4j.randn(DataType.DOUBLE, 4);
        INDArray gradOut = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0});
        double decay = 0.9;

        INDArray dModel = Nd4j.create(DataType.DOUBLE, 4);
        INDArray dShadow = Nd4j.create(DataType.DOUBLE, 4);

        DynamicCustomOp op = DynamicCustomOp.builder("ema_update_bp")
                .addInputs(model, shadow, gradOut)
                .addOutputs(dModel, dShadow)
                .addFloatingPointArguments(decay)
                .build();
        Nd4j.getExecutioner().exec(op);

        INDArray expectedDModel = gradOut.mul(1.0 - decay);
        INDArray expectedDShadow = gradOut.mul(decay);

        assertEquals(expectedDModel, dModel);
        assertEquals(expectedDShadow, dShadow);
    }

    // ==================== Center and Sharpen ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCenterAndSharpenBasic(Nd4jBackend backend) {
        // output = softmax((input - center) / temperature)
        int batch = 2;
        int features = 4;
        double temperature = 1.0;

        INDArray input = Nd4j.create(new double[][]{
                {1.0, 2.0, 3.0, 4.0},
                {4.0, 3.0, 2.0, 1.0}
        });
        INDArray center = Nd4j.zeros(DataType.DOUBLE, features);
        INDArray output = Nd4j.create(DataType.DOUBLE, batch, features);

        DynamicCustomOp op = DynamicCustomOp.builder("center_and_sharpen")
                .addInputs(input, center)
                .addOutputs(output)
                .addFloatingPointArguments(temperature)
                .build();
        Nd4j.getExecutioner().exec(op);

        // With center=0 and temp=1, this is just softmax
        // Verify rows sum to 1
        for (int b = 0; b < batch; b++) {
            double rowSum = output.getRow(b).sumNumber().doubleValue();
            assertEquals(1.0, rowSum, 1e-5, "Softmax row should sum to 1");
        }

        // Verify all values are positive
        assertTrue(output.minNumber().doubleValue() > 0, "All softmax values should be positive");

        // Verify the max element has the highest probability
        assertEquals(3, Nd4j.argMax(output.getRow(0), 0).getInt(0),
                "Largest input should have highest probability");
        assertEquals(0, Nd4j.argMax(output.getRow(1), 0).getInt(0),
                "Largest input should have highest probability");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCenterAndSharpenWithCenter(Nd4jBackend backend) {
        // Centering should shift the distribution
        int batch = 1;
        int features = 3;
        double temperature = 1.0;

        INDArray input = Nd4j.create(new double[][]{{3.0, 3.0, 3.0}});
        INDArray center = Nd4j.create(new double[]{3.0, 3.0, 3.0});
        INDArray output = Nd4j.create(DataType.DOUBLE, batch, features);

        DynamicCustomOp op = DynamicCustomOp.builder("center_and_sharpen")
                .addInputs(input, center)
                .addOutputs(output)
                .addFloatingPointArguments(temperature)
                .build();
        Nd4j.getExecutioner().exec(op);

        // input - center = [0,0,0], softmax([0,0,0]) = [1/3, 1/3, 1/3]
        INDArray expected = Nd4j.create(new double[][]{{1.0 / 3, 1.0 / 3, 1.0 / 3}});
        for (int i = 0; i < features; i++) {
            assertEquals(expected.getDouble(0, i), output.getDouble(0, i), 1e-5);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCenterAndSharpenLowTemperature(Nd4jBackend backend) {
        // Low temperature should make distribution sharper (more peaked)
        int batch = 1;
        int features = 4;

        INDArray input = Nd4j.create(new double[][]{{1.0, 2.0, 3.0, 4.0}});
        INDArray center = Nd4j.zeros(DataType.DOUBLE, features);

        INDArray outputWarm = Nd4j.create(DataType.DOUBLE, batch, features);
        INDArray outputCold = Nd4j.create(DataType.DOUBLE, batch, features);

        DynamicCustomOp opWarm = DynamicCustomOp.builder("center_and_sharpen")
                .addInputs(input, center)
                .addOutputs(outputWarm)
                .addFloatingPointArguments(1.0)
                .build();
        Nd4j.getExecutioner().exec(opWarm);

        DynamicCustomOp opCold = DynamicCustomOp.builder("center_and_sharpen")
                .addInputs(input, center)
                .addOutputs(outputCold)
                .addFloatingPointArguments(0.1)
                .build();
        Nd4j.getExecutioner().exec(opCold);

        // Lower temperature should give higher max probability
        assertTrue(outputCold.maxNumber().doubleValue() > outputWarm.maxNumber().doubleValue(),
                "Lower temperature should produce sharper distribution");
    }

    // ==================== Center and Sharpen Backward ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCenterAndSharpenBpShapes(Nd4jBackend backend) {
        int batch = 2;
        int features = 4;
        double temperature = 0.5;

        INDArray input = Nd4j.randn(DataType.DOUBLE, batch, features);
        INDArray center = Nd4j.randn(DataType.DOUBLE, features);
        INDArray gradOut = Nd4j.randn(DataType.DOUBLE, batch, features);

        INDArray dInput = Nd4j.create(DataType.DOUBLE, batch, features);
        INDArray dCenter = Nd4j.create(DataType.DOUBLE, features);

        DynamicCustomOp op = DynamicCustomOp.builder("center_and_sharpen_bp")
                .addInputs(input, center, gradOut)
                .addOutputs(dInput, dCenter)
                .addFloatingPointArguments(temperature)
                .build();
        Nd4j.getExecutioner().exec(op);

        assertArrayEquals(new long[]{batch, features}, dInput.shape());
        assertArrayEquals(new long[]{features}, dCenter.shape());
        assertFalse(dInput.isNaN().any(), "dInput should not contain NaN");
        assertFalse(dCenter.isNaN().any(), "dCenter should not contain NaN");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCenterAndSharpenBpNumericalGradient(Nd4jBackend backend) {
        // Numerical gradient check for dL/dInput
        int batch = 2;
        int features = 3;
        double temperature = 0.5;
        double eps = 1e-5;

        INDArray input = Nd4j.randn(DataType.DOUBLE, batch, features);
        INDArray center = Nd4j.randn(DataType.DOUBLE, features);
        // Use random gradOut — uniform gradOut is in the null space of softmax Jacobian
        INDArray gradOut = Nd4j.randn(DataType.DOUBLE, batch, features);

        // Analytical gradient
        INDArray dInput = Nd4j.create(DataType.DOUBLE, batch, features);
        INDArray dCenter = Nd4j.create(DataType.DOUBLE, features);
        DynamicCustomOp bpOp = DynamicCustomOp.builder("center_and_sharpen_bp")
                .addInputs(input, center, gradOut)
                .addOutputs(dInput, dCenter)
                .addFloatingPointArguments(temperature)
                .build();
        Nd4j.getExecutioner().exec(bpOp);

        // Numerical gradient for each input element
        for (int b = 0; b < batch; b++) {
            for (int f = 0; f < features; f++) {
                double orig = input.getDouble(b, f);

                input.putScalar(b, f, orig + eps);
                INDArray outPlus = Nd4j.create(DataType.DOUBLE, batch, features);
                DynamicCustomOp fwdPlus = DynamicCustomOp.builder("center_and_sharpen")
                        .addInputs(input, center)
                        .addOutputs(outPlus)
                        .addFloatingPointArguments(temperature)
                        .build();
                Nd4j.getExecutioner().exec(fwdPlus);

                input.putScalar(b, f, orig - eps);
                INDArray outMinus = Nd4j.create(DataType.DOUBLE, batch, features);
                DynamicCustomOp fwdMinus = DynamicCustomOp.builder("center_and_sharpen")
                        .addInputs(input, center)
                        .addOutputs(outMinus)
                        .addFloatingPointArguments(temperature)
                        .build();
                Nd4j.getExecutioner().exec(fwdMinus);

                input.putScalar(b, f, orig);

                // dL/dx_bf = sum over outputs of gradOut * (outPlus - outMinus) / (2*eps)
                double numericalGrad = outPlus.sub(outMinus).mul(gradOut).sumNumber().doubleValue() / (2 * eps);
                double analyticalGrad = dInput.getDouble(b, f);

                // Relaxed tolerance: CUDA softmax uses float32 intermediates even for double
                assertEquals(numericalGrad, analyticalGrad, 5e-2,
                        String.format("Gradient mismatch at [%d,%d]: numerical=%.6f, analytical=%.6f",
                                b, f, numericalGrad, analyticalGrad));
            }
        }
    }

    // ==================== Contrastive Loss ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testContrastiveLossIdentical(Nd4jBackend backend) {
        // When image and text embeddings are identical unit vectors, loss should be low
        int batch = 4;
        int embedDim = 8;
        double temperature = 1.0;

        // Create orthogonal unit vectors — pad with zeros if embedDim > batch
        INDArray embeddings = Nd4j.eye(batch);
        if (embedDim > batch) {
            embeddings = Nd4j.hstack(embeddings, Nd4j.zeros(batch, embedDim - batch));
        }

        DynamicCustomOp op = DynamicCustomOp.builder("contrastive_loss")
                .addInputs(embeddings, embeddings)
                .addFloatingPointArguments(temperature)
                .build();
        Nd4j.getExecutioner().exec(op);
        INDArray output = op.getOutputArgument(0);

        // With identical embeddings, diagonal has max similarity
        // Loss should be close to log(batch) for each row/col
        double loss = output.getDouble(0);
        assertFalse(Double.isNaN(loss), "Loss should not be NaN");
        assertTrue(loss >= 0, "Loss should be non-negative");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testContrastiveLossManual(Nd4jBackend backend) {
        // Manual computation with small batch
        int batch = 2;
        int embedDim = 3;
        double temperature = 1.0;

        INDArray imageEmb = Nd4j.create(new double[][]{
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0}
        });
        INDArray textEmb = Nd4j.create(new double[][]{
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0}
        });

        DynamicCustomOp op = DynamicCustomOp.builder("contrastive_loss")
                .addInputs(imageEmb, textEmb)
                .addFloatingPointArguments(temperature)
                .build();
        Nd4j.getExecutioner().exec(op);
        INDArray output = op.getOutputArgument(0);

        // sim = [[1,0],[0,1]] * temp = [[1,0],[0,1]]
        // row CE for row 0: -sim[0,0] + log(exp(1)+exp(0)) = -1 + log(e+1) ≈ -1 + 1.3133 = 0.3133
        // row CE for row 1: -sim[1,1] + log(exp(0)+exp(1)) = same = 0.3133
        // col CE: same by symmetry
        // loss = (sum_row + sum_col) / (2*batch) = (0.6266 + 0.6266) / 4 = 0.3133
        double expectedLoss = -1.0 + Math.log(Math.exp(1.0) + Math.exp(0.0));
        // average over rows and cols: same value
        assertEquals(expectedLoss, output.getDouble(0), 1e-4,
                "Contrastive loss should match manual computation");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testContrastiveLossTemperatureEffect(Nd4jBackend backend) {
        int batch = 4;
        int embedDim = 8;

        INDArray imageEmb = Nd4j.randn(DataType.DOUBLE, batch, embedDim);
        INDArray textEmb = Nd4j.randn(DataType.DOUBLE, batch, embedDim);

        DynamicCustomOp opLow = DynamicCustomOp.builder("contrastive_loss")
                .addInputs(imageEmb, textEmb)
                .addFloatingPointArguments(0.1)
                .build();
        Nd4j.getExecutioner().exec(opLow);
        INDArray lossLowTemp = opLow.getOutputArgument(0);

        DynamicCustomOp opHigh = DynamicCustomOp.builder("contrastive_loss")
                .addInputs(imageEmb, textEmb)
                .addFloatingPointArguments(10.0)
                .build();
        Nd4j.getExecutioner().exec(opHigh);
        INDArray lossHighTemp = opHigh.getOutputArgument(0);

        assertFalse(Double.isNaN(lossLowTemp.getDouble(0)), "Low temp loss should not be NaN");
        assertFalse(Double.isNaN(lossHighTemp.getDouble(0)), "High temp loss should not be NaN");
        // Both should be valid non-negative losses
        assertTrue(lossLowTemp.getDouble(0) >= 0, "Loss should be non-negative");
        assertTrue(lossHighTemp.getDouble(0) >= 0, "Loss should be non-negative");
    }

    // ==================== Contrastive Loss Backward ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testContrastiveLossBpShapes(Nd4jBackend backend) {
        int batch = 4;
        int embedDim = 8;
        double temperature = 1.0;

        INDArray imageEmb = Nd4j.randn(DataType.DOUBLE, batch, embedDim);
        INDArray textEmb = Nd4j.randn(DataType.DOUBLE, batch, embedDim);

        INDArray dImage = Nd4j.create(DataType.DOUBLE, batch, embedDim);
        INDArray dText = Nd4j.create(DataType.DOUBLE, batch, embedDim);

        DynamicCustomOp op = DynamicCustomOp.builder("contrastive_loss_grad")
                .addInputs(imageEmb, textEmb)
                .addOutputs(dImage, dText)
                .addFloatingPointArguments(temperature)
                .build();
        Nd4j.getExecutioner().exec(op);

        assertArrayEquals(new long[]{batch, embedDim}, dImage.shape());
        assertArrayEquals(new long[]{batch, embedDim}, dText.shape());
        assertFalse(dImage.isNaN().any(), "dImage should not contain NaN");
        assertFalse(dText.isNaN().any(), "dText should not contain NaN");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testContrastiveLossBpNumericalGradient(Nd4jBackend backend) {
        int batch = 2;
        int embedDim = 3;
        double temperature = 1.0;
        double eps = 1e-5;

        INDArray imageEmb = Nd4j.randn(DataType.DOUBLE, batch, embedDim);
        INDArray textEmb = Nd4j.randn(DataType.DOUBLE, batch, embedDim);

        // Analytical gradients
        INDArray dImage = Nd4j.create(DataType.DOUBLE, batch, embedDim);
        INDArray dText = Nd4j.create(DataType.DOUBLE, batch, embedDim);
        DynamicCustomOp bpOp = DynamicCustomOp.builder("contrastive_loss_grad")
                .addInputs(imageEmb, textEmb)
                .addOutputs(dImage, dText)
                .addFloatingPointArguments(temperature)
                .build();
        Nd4j.getExecutioner().exec(bpOp);

        // Numerical gradient for imageEmb
        for (int b = 0; b < batch; b++) {
            for (int d = 0; d < embedDim; d++) {
                double orig = imageEmb.getDouble(b, d);

                imageEmb.putScalar(b, d, orig + eps);
                DynamicCustomOp fwdPlus = DynamicCustomOp.builder("contrastive_loss")
                        .addInputs(imageEmb, textEmb)
                        .addFloatingPointArguments(temperature)
                        .build();
                Nd4j.getExecutioner().exec(fwdPlus);
                INDArray lossPlus = fwdPlus.getOutputArgument(0);

                imageEmb.putScalar(b, d, orig - eps);
                DynamicCustomOp fwdMinus = DynamicCustomOp.builder("contrastive_loss")
                        .addInputs(imageEmb, textEmb)
                        .addFloatingPointArguments(temperature)
                        .build();
                Nd4j.getExecutioner().exec(fwdMinus);
                INDArray lossMinus = fwdMinus.getOutputArgument(0);

                imageEmb.putScalar(b, d, orig);

                double numericalGrad = (lossPlus.getDouble(0) - lossMinus.getDouble(0)) / (2 * eps);
                double analyticalGrad = dImage.getDouble(b, d);

                assertEquals(numericalGrad, analyticalGrad, 2e-2,
                        String.format("Image grad mismatch at [%d,%d]: numerical=%.6f, analytical=%.6f",
                                b, d, numericalGrad, analyticalGrad));
            }
        }
    }

    // ==================== Two-Way Cross Attention ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTwoWayCrossAttentionBasic(Nd4jBackend backend) {
        int batch = 1;
        int tokenSeq = 4;
        int imageSeq = 6;
        int dim = 8;

        INDArray tokenQ = Nd4j.randn(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray tokenK = Nd4j.randn(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray tokenV = Nd4j.randn(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray imageQ = Nd4j.randn(DataType.DOUBLE, batch, imageSeq, dim);
        INDArray imageK = Nd4j.randn(DataType.DOUBLE, batch, imageSeq, dim);
        INDArray imageV = Nd4j.randn(DataType.DOUBLE, batch, imageSeq, dim);

        INDArray tokenOut = Nd4j.create(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray imageOut = Nd4j.create(DataType.DOUBLE, batch, imageSeq, dim);

        double scale = 1.0 / Math.sqrt(dim);

        DynamicCustomOp op = DynamicCustomOp.builder("two_way_cross_attention")
                .addInputs(tokenQ, tokenK, tokenV, imageQ, imageK, imageV)
                .addOutputs(tokenOut, imageOut)
                .addFloatingPointArguments(scale)
                .build();
        Nd4j.getExecutioner().exec(op);

        // Check output shapes
        assertArrayEquals(new long[]{batch, tokenSeq, dim}, tokenOut.shape());
        assertArrayEquals(new long[]{batch, imageSeq, dim}, imageOut.shape());

        // Check no NaN
        assertFalse(tokenOut.isNaN().any(), "Token output should not contain NaN");
        assertFalse(imageOut.isNaN().any(), "Image output should not contain NaN");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTwoWayCrossAttentionManual(Nd4jBackend backend) {
        // Manual verification: tokenOut = softmax(tokenQ @ imageK^T * scale) @ imageV
        // Use 2D inputs to test rank-2 kernel path
        int tokenSeq = 2;
        int imageSeq = 3;
        int dim = 4;

        INDArray tokenQ = Nd4j.randn(DataType.FLOAT, tokenSeq, dim);
        INDArray tokenK = Nd4j.randn(DataType.FLOAT, tokenSeq, dim);
        INDArray tokenV = Nd4j.randn(DataType.FLOAT, tokenSeq, dim);
        INDArray imageQ = Nd4j.randn(DataType.FLOAT, imageSeq, dim);
        INDArray imageK = Nd4j.randn(DataType.FLOAT, imageSeq, dim);
        INDArray imageV = Nd4j.randn(DataType.FLOAT, imageSeq, dim);

        double scale = 1.0 / Math.sqrt(dim);

        INDArray tokenOut = Nd4j.create(DataType.FLOAT, tokenSeq, dim);
        INDArray imageOut = Nd4j.create(DataType.FLOAT, imageSeq, dim);

        DynamicCustomOp op = DynamicCustomOp.builder("two_way_cross_attention")
                .addInputs(tokenQ, tokenK, tokenV, imageQ, imageK, imageV)
                .addOutputs(tokenOut, imageOut)
                .addFloatingPointArguments(scale)
                .build();
        Nd4j.getExecutioner().exec(op);

        // Verify no Inf or NaN in outputs
        assertFalse(tokenOut.isInfinite().any(), "Token output should not contain Inf");
        assertFalse(tokenOut.isNaN().any(), "Token output should not contain NaN");
        assertFalse(imageOut.isInfinite().any(), "Image output should not contain Inf");
        assertFalse(imageOut.isNaN().any(), "Image output should not contain NaN");

        // Compute expected output using Java doubles to avoid gemm's built-in Inf check
        // tokenOutput = softmax(tokenQ @ imageK^T * scale) @ imageV
        for (int i = 0; i < tokenSeq; i++) {
            // Compute logits[j] = dot(tokenQ[i], imageK[j]) * scale
            double[] logits = new double[imageSeq];
            double maxLogit = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < imageSeq; j++) {
                double dot = 0;
                for (int k = 0; k < dim; k++) {
                    dot += tokenQ.getDouble(i, k) * imageK.getDouble(j, k);
                }
                logits[j] = dot * scale;
                maxLogit = Math.max(maxLogit, logits[j]);
            }
            // Softmax
            double sumE = 0;
            double[] weights = new double[imageSeq];
            for (int j = 0; j < imageSeq; j++) {
                weights[j] = Math.exp(logits[j] - maxLogit);
                sumE += weights[j];
            }
            for (int j = 0; j < imageSeq; j++) {
                weights[j] /= sumE;
            }
            // Output[i][d] = sum_j(weights[j] * imageV[j][d])
            for (int d = 0; d < dim; d++) {
                double acc = 0;
                for (int j = 0; j < imageSeq; j++) {
                    acc += weights[j] * imageV.getDouble(j, d);
                }
                assertEquals(acc, tokenOut.getDouble(i, d), 1e-3,
                        String.format("Token output mismatch at [%d,%d]", i, d));
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTwoWayCrossAttentionBatch(Nd4jBackend backend) {
        // Test with larger batch
        int batch = 3;
        int tokenSeq = 4;
        int imageSeq = 8;
        int dim = 16;

        INDArray tokenQ = Nd4j.randn(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray tokenK = Nd4j.randn(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray tokenV = Nd4j.randn(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray imageQ = Nd4j.randn(DataType.DOUBLE, batch, imageSeq, dim);
        INDArray imageK = Nd4j.randn(DataType.DOUBLE, batch, imageSeq, dim);
        INDArray imageV = Nd4j.randn(DataType.DOUBLE, batch, imageSeq, dim);

        INDArray tokenOut = Nd4j.create(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray imageOut = Nd4j.create(DataType.DOUBLE, batch, imageSeq, dim);

        double scale = 1.0 / Math.sqrt(dim);

        DynamicCustomOp op = DynamicCustomOp.builder("two_way_cross_attention")
                .addInputs(tokenQ, tokenK, tokenV, imageQ, imageK, imageV)
                .addOutputs(tokenOut, imageOut)
                .addFloatingPointArguments(scale)
                .build();
        Nd4j.getExecutioner().exec(op);

        assertArrayEquals(new long[]{batch, tokenSeq, dim}, tokenOut.shape());
        assertArrayEquals(new long[]{batch, imageSeq, dim}, imageOut.shape());
        assertFalse(tokenOut.isNaN().any(), "Token output should not contain NaN");
        assertFalse(imageOut.isNaN().any(), "Image output should not contain NaN");
    }

    // ==================== Two-Way Cross Attention Backward ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTwoWayCrossAttentionBpShapes(Nd4jBackend backend) {
        int batch = 1;
        int tokenSeq = 3;
        int imageSeq = 4;
        int dim = 8;
        double scale = 1.0 / Math.sqrt(dim);

        INDArray tokenQ = Nd4j.randn(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray tokenK = Nd4j.randn(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray tokenV = Nd4j.randn(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray imageQ = Nd4j.randn(DataType.DOUBLE, batch, imageSeq, dim);
        INDArray imageK = Nd4j.randn(DataType.DOUBLE, batch, imageSeq, dim);
        INDArray imageV = Nd4j.randn(DataType.DOUBLE, batch, imageSeq, dim);
        INDArray gradTokenOut = Nd4j.randn(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray gradImageOut = Nd4j.randn(DataType.DOUBLE, batch, imageSeq, dim);

        INDArray dTokenQ = Nd4j.create(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray dTokenK = Nd4j.create(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray dTokenV = Nd4j.create(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray dImageQ = Nd4j.create(DataType.DOUBLE, batch, imageSeq, dim);
        INDArray dImageK = Nd4j.create(DataType.DOUBLE, batch, imageSeq, dim);
        INDArray dImageV = Nd4j.create(DataType.DOUBLE, batch, imageSeq, dim);

        DynamicCustomOp op = DynamicCustomOp.builder("two_way_cross_attention_bp")
                .addInputs(tokenQ, tokenK, tokenV, imageQ, imageK, imageV,
                        gradTokenOut, gradImageOut)
                .addOutputs(dTokenQ, dTokenK, dTokenV, dImageQ, dImageK, dImageV)
                .addFloatingPointArguments(scale)
                .build();
        Nd4j.getExecutioner().exec(op);

        assertArrayEquals(new long[]{batch, tokenSeq, dim}, dTokenQ.shape());
        assertArrayEquals(new long[]{batch, tokenSeq, dim}, dTokenK.shape());
        assertArrayEquals(new long[]{batch, tokenSeq, dim}, dTokenV.shape());
        assertArrayEquals(new long[]{batch, imageSeq, dim}, dImageQ.shape());
        assertArrayEquals(new long[]{batch, imageSeq, dim}, dImageK.shape());
        assertArrayEquals(new long[]{batch, imageSeq, dim}, dImageV.shape());

        assertFalse(dTokenQ.isNaN().any(), "dTokenQ should not contain NaN");
        assertFalse(dTokenK.isNaN().any(), "dTokenK should not contain NaN");
        assertFalse(dTokenV.isNaN().any(), "dTokenV should not contain NaN");
        assertFalse(dImageQ.isNaN().any(), "dImageQ should not contain NaN");
        assertFalse(dImageK.isNaN().any(), "dImageK should not contain NaN");
        assertFalse(dImageV.isNaN().any(), "dImageV should not contain NaN");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTwoWayCrossAttentionBpNumericalGradientTokenQ(Nd4jBackend backend) {
        // Numerical gradient check for dTokenQ
        int batch = 1;
        int tokenSeq = 2;
        int imageSeq = 2;
        int dim = 3;
        double scale = 1.0 / Math.sqrt(dim);
        double eps = 1e-5;

        INDArray tokenQ = Nd4j.randn(DataType.DOUBLE, batch, tokenSeq, dim).mul(0.5);
        INDArray tokenK = Nd4j.randn(DataType.DOUBLE, batch, tokenSeq, dim).mul(0.5);
        INDArray tokenV = Nd4j.randn(DataType.DOUBLE, batch, tokenSeq, dim).mul(0.5);
        INDArray imageQ = Nd4j.randn(DataType.DOUBLE, batch, imageSeq, dim).mul(0.5);
        INDArray imageK = Nd4j.randn(DataType.DOUBLE, batch, imageSeq, dim).mul(0.5);
        INDArray imageV = Nd4j.randn(DataType.DOUBLE, batch, imageSeq, dim).mul(0.5);

        // Use sum of all outputs as scalar loss for gradient checking
        INDArray gradTokenOut = Nd4j.ones(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray gradImageOut = Nd4j.ones(DataType.DOUBLE, batch, imageSeq, dim);

        // Analytical gradients
        INDArray dTokenQ = Nd4j.create(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray dTokenK = Nd4j.create(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray dTokenV = Nd4j.create(DataType.DOUBLE, batch, tokenSeq, dim);
        INDArray dImageQ = Nd4j.create(DataType.DOUBLE, batch, imageSeq, dim);
        INDArray dImageK = Nd4j.create(DataType.DOUBLE, batch, imageSeq, dim);
        INDArray dImageV = Nd4j.create(DataType.DOUBLE, batch, imageSeq, dim);

        DynamicCustomOp bpOp = DynamicCustomOp.builder("two_way_cross_attention_bp")
                .addInputs(tokenQ, tokenK, tokenV, imageQ, imageK, imageV,
                        gradTokenOut, gradImageOut)
                .addOutputs(dTokenQ, dTokenK, dTokenV, dImageQ, dImageK, dImageV)
                .addFloatingPointArguments(scale)
                .build();
        Nd4j.getExecutioner().exec(bpOp);

        // Numerical gradient for tokenQ
        for (int s = 0; s < tokenSeq; s++) {
            for (int d = 0; d < dim; d++) {
                double orig = tokenQ.getDouble(0, s, d);

                tokenQ.putScalar(new int[]{0, s, d}, orig + eps);
                INDArray tokOutPlus = Nd4j.create(DataType.DOUBLE, batch, tokenSeq, dim);
                INDArray imgOutPlus = Nd4j.create(DataType.DOUBLE, batch, imageSeq, dim);
                DynamicCustomOp fwdPlus = DynamicCustomOp.builder("two_way_cross_attention")
                        .addInputs(tokenQ, tokenK, tokenV, imageQ, imageK, imageV)
                        .addOutputs(tokOutPlus, imgOutPlus)
                        .addFloatingPointArguments(scale)
                        .build();
                Nd4j.getExecutioner().exec(fwdPlus);
                double lossPlus = tokOutPlus.sumNumber().doubleValue() + imgOutPlus.sumNumber().doubleValue();

                tokenQ.putScalar(new int[]{0, s, d}, orig - eps);
                INDArray tokOutMinus = Nd4j.create(DataType.DOUBLE, batch, tokenSeq, dim);
                INDArray imgOutMinus = Nd4j.create(DataType.DOUBLE, batch, imageSeq, dim);
                DynamicCustomOp fwdMinus = DynamicCustomOp.builder("two_way_cross_attention")
                        .addInputs(tokenQ, tokenK, tokenV, imageQ, imageK, imageV)
                        .addOutputs(tokOutMinus, imgOutMinus)
                        .addFloatingPointArguments(scale)
                        .build();
                Nd4j.getExecutioner().exec(fwdMinus);
                double lossMinus = tokOutMinus.sumNumber().doubleValue() + imgOutMinus.sumNumber().doubleValue();

                tokenQ.putScalar(new int[]{0, s, d}, orig);

                double numericalGrad = (lossPlus - lossMinus) / (2 * eps);
                double analyticalGrad = dTokenQ.getDouble(0, s, d);

                // Relaxed tolerance: attention weights computed in float32 even for double inputs
                assertEquals(numericalGrad, analyticalGrad, 5e-2,
                        String.format("dTokenQ mismatch at [0,%d,%d]: numerical=%.6f, analytical=%.6f",
                                s, d, numericalGrad, analyticalGrad));
            }
        }
    }

    // ==================== Float32 type tests ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmaUpdateFloat32(Nd4jBackend backend) {
        INDArray model = Nd4j.randn(DataType.FLOAT, 10);
        INDArray shadow = Nd4j.randn(DataType.FLOAT, 10);
        INDArray output = Nd4j.create(DataType.FLOAT, 10);

        DynamicCustomOp op = DynamicCustomOp.builder("ema_update")
                .addInputs(model, shadow)
                .addOutputs(output)
                .addFloatingPointArguments(0.999)
                .build();
        Nd4j.getExecutioner().exec(op);

        assertFalse(output.isNaN().any(), "Float32 EMA output should not contain NaN");
        assertArrayEquals(model.shape(), output.shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCenterAndSharpenFloat32(Nd4jBackend backend) {
        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 16);
        INDArray center = Nd4j.randn(DataType.FLOAT, 16);
        INDArray output = Nd4j.create(DataType.FLOAT, 4, 16);

        DynamicCustomOp op = DynamicCustomOp.builder("center_and_sharpen")
                .addInputs(input, center)
                .addOutputs(output)
                .addFloatingPointArguments(0.07)
                .build();
        Nd4j.getExecutioner().exec(op);

        assertFalse(output.isNaN().any(), "Float32 center_and_sharpen should not contain NaN");
        // Rows should sum to 1
        for (int b = 0; b < 4; b++) {
            assertEquals(1.0, output.getRow(b).sumNumber().doubleValue(), 1e-4);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testContrastiveLossFloat32(Nd4jBackend backend) {
        INDArray imageEmb = Nd4j.randn(DataType.FLOAT, 4, 8);
        INDArray textEmb = Nd4j.randn(DataType.FLOAT, 4, 8);

        DynamicCustomOp op = DynamicCustomOp.builder("contrastive_loss")
                .addInputs(imageEmb, textEmb)
                .addFloatingPointArguments(1.0)
                .build();
        Nd4j.getExecutioner().exec(op);
        INDArray output = op.getOutputArgument(0);

        assertFalse(Double.isNaN(output.getDouble(0)), "Float32 contrastive loss should not be NaN");
        assertTrue(output.getDouble(0) >= 0, "Loss should be non-negative");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTwoWayCrossAttentionFloat32(Nd4jBackend backend) {
        int batch = 2, tokenSeq = 4, imageSeq = 6, dim = 16;

        INDArray tokenQ = Nd4j.randn(DataType.FLOAT, batch, tokenSeq, dim);
        INDArray tokenK = Nd4j.randn(DataType.FLOAT, batch, tokenSeq, dim);
        INDArray tokenV = Nd4j.randn(DataType.FLOAT, batch, tokenSeq, dim);
        INDArray imageQ = Nd4j.randn(DataType.FLOAT, batch, imageSeq, dim);
        INDArray imageK = Nd4j.randn(DataType.FLOAT, batch, imageSeq, dim);
        INDArray imageV = Nd4j.randn(DataType.FLOAT, batch, imageSeq, dim);

        INDArray tokenOut = Nd4j.create(DataType.FLOAT, batch, tokenSeq, dim);
        INDArray imageOut = Nd4j.create(DataType.FLOAT, batch, imageSeq, dim);

        DynamicCustomOp op = DynamicCustomOp.builder("two_way_cross_attention")
                .addInputs(tokenQ, tokenK, tokenV, imageQ, imageK, imageV)
                .addOutputs(tokenOut, imageOut)
                .addFloatingPointArguments(1.0 / Math.sqrt(dim))
                .build();
        Nd4j.getExecutioner().exec(op);

        assertFalse(tokenOut.isNaN().any(), "Float32 token output should not contain NaN");
        assertFalse(imageOut.isNaN().any(), "Float32 image output should not contain NaN");
    }
}
