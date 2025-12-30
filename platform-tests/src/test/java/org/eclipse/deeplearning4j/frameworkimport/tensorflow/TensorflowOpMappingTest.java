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

package org.eclipse.deeplearning4j.frameworkimport.tensorflow;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.samediff.frameworkimport.tensorflow.definitions.TensorflowOpDeclarationsKt;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for TensorFlow op mappings.
 * Tests that the newly added TensorFlow ops are correctly mapped to nd4j ops.
 */
@Tag(TagNames.TENSORFLOW)
public class TensorflowOpMappingTest {

    @BeforeAll
    public static void setup() {
        // Initialize nd4j
        Nd4j.scalar(1.0);
    }

    /**
     * Test that all the newly added op mappings exist in the registry.
     */
    @Test
    public void testOpMappingsExist() {
        var registry = TensorflowOpDeclarationsKt.registry();

        // Test identity-type ops
        assertTrue(registry.hasMappingOpProcess("Copy"), "Copy mapping should exist");
        assertTrue(registry.hasMappingOpProcess("Snapshot"), "Snapshot mapping should exist");
        assertTrue(registry.hasMappingOpProcess("PreventGradient"), "PreventGradient mapping should exist");
        assertTrue(registry.hasMappingOpProcess("StopGradient"), "StopGradient mapping should exist");
        assertTrue(registry.hasMappingOpProcess("Identity"), "Identity mapping should exist");
        assertTrue(registry.hasMappingOpProcess("CopyHost"), "CopyHost mapping should exist");
        assertTrue(registry.hasMappingOpProcess("DeepCopy"), "DeepCopy mapping should exist");

        // Test safe math ops
        assertTrue(registry.hasMappingOpProcess("Xdivy"), "Xdivy mapping should exist");
        assertTrue(registry.hasMappingOpProcess("Xlogy"), "Xlogy mapping should exist");
        assertTrue(registry.hasMappingOpProcess("Xlog1py"), "Xlog1py mapping should exist");

        // Test gradient ops
        assertTrue(registry.hasMappingOpProcess("SigmoidGrad"), "SigmoidGrad mapping should exist");
        assertTrue(registry.hasMappingOpProcess("TanhGrad"), "TanhGrad mapping should exist");
        assertTrue(registry.hasMappingOpProcess("SoftmaxGrad"), "SoftmaxGrad mapping should exist");
        assertTrue(registry.hasMappingOpProcess("SoftplusGrad"), "SoftplusGrad mapping should exist");
        assertTrue(registry.hasMappingOpProcess("SoftsignGrad"), "SoftsignGrad mapping should exist");
        assertTrue(registry.hasMappingOpProcess("LeakyReluGrad"), "LeakyReluGrad mapping should exist");
        assertTrue(registry.hasMappingOpProcess("LRNGrad"), "LRNGrad mapping should exist");
        assertTrue(registry.hasMappingOpProcess("ReluGrad"), "ReluGrad mapping should exist");
        assertTrue(registry.hasMappingOpProcess("EluGrad"), "EluGrad mapping should exist");
        assertTrue(registry.hasMappingOpProcess("Relu6Grad"), "Relu6Grad mapping should exist");
        assertTrue(registry.hasMappingOpProcess("SeluGrad"), "SeluGrad mapping should exist");

        // Test array ops
        assertTrue(registry.hasMappingOpProcess("ReverseV2"), "ReverseV2 mapping should exist");
        assertTrue(registry.hasMappingOpProcess("MatrixBandPart"), "MatrixBandPart mapping should exist");

        // Test linear algebra ops
        assertTrue(registry.hasMappingOpProcess("Qr"), "Qr mapping should exist");
        assertTrue(registry.hasMappingOpProcess("Eig"), "Eig mapping should exist");

        // Test Einsum
        assertTrue(registry.hasMappingOpProcess("Einsum"), "Einsum mapping should exist");
    }

    /**
     * Test identity op execution using SameDiff.
     */
    @Test
    public void testIdentityOp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});

        var x = sd.var("x", input);
        var result = sd.identity(x);

        INDArray output = result.eval();
        assertEquals(input, output, "Identity op should return same array");
    }

    /**
     * Test sigmoid gradient backprop.
     */
    @Test
    public void testSigmoidBp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.create(new float[]{0, 1, -1, 2}, new int[]{2, 2});

        var x = sd.var("x", input);
        var sigmoid = sd.nn().sigmoid(x);
        var loss = sigmoid.sum();

        // Compute gradients
        sd.setLossVariables(loss);
        sd.createGradFunction();

        var gradMap = sd.calculateGradients(null, "x");
        INDArray grad = gradMap.get("x");

        // Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
        assertNotNull(grad, "Gradient should not be null");
        assertEquals(input.shape(), grad.shape(), "Gradient shape should match input");
    }

    /**
     * Test tanh gradient backprop.
     */
    @Test
    public void testTanhBp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.create(new float[]{0, 1, -1, 2}, new int[]{2, 2});

        var x = sd.var("x", input);
        var tanh = sd.math().tanh(x);
        var loss = tanh.sum();

        sd.setLossVariables(loss);
        sd.createGradFunction();

        var gradMap = sd.calculateGradients(null, "x");
        INDArray grad = gradMap.get("x");

        // Tanh gradient: 1 - tanh(x)^2
        assertNotNull(grad, "Gradient should not be null");
        assertEquals(input.shape(), grad.shape(), "Gradient shape should match input");
    }

    /**
     * Test ReLU gradient backprop.
     */
    @Test
    public void testReluBp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.create(new float[]{-1, 0, 1, 2}, new int[]{2, 2});

        var x = sd.var("x", input);
        var relu = sd.nn().relu(x, 0);
        var loss = relu.sum();

        sd.setLossVariables(loss);
        sd.createGradFunction();

        var gradMap = sd.calculateGradients(null, "x");
        INDArray grad = gradMap.get("x");

        // ReLU gradient: 0 where x < 0, 1 where x >= 0
        assertNotNull(grad, "Gradient should not be null");
        assertEquals(0.0, grad.getDouble(0), 1e-5, "Gradient should be 0 for negative input");
        assertEquals(1.0, grad.getDouble(2), 1e-5, "Gradient should be 1 for positive input");
    }

    /**
     * Test ELU gradient backprop.
     */
    @Test
    public void testEluBp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.create(new float[]{-1, 0, 1, 2}, new int[]{2, 2});

        var x = sd.var("x", input);
        var elu = sd.nn().elu(x);
        var loss = elu.sum();

        sd.setLossVariables(loss);
        sd.createGradFunction();

        var gradMap = sd.calculateGradients(null, "x");
        INDArray grad = gradMap.get("x");

        assertNotNull(grad, "Gradient should not be null");
        assertEquals(input.shape(), grad.shape(), "Gradient shape should match input");
    }

    /**
     * Test SELU gradient backprop.
     */
    @Test
    public void testSeluBp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.create(new float[]{-1, 0, 1, 2}, new int[]{2, 2});

        var x = sd.var("x", input);
        var selu = sd.nn().selu(x);
        var loss = selu.sum();

        sd.setLossVariables(loss);
        sd.createGradFunction();

        var gradMap = sd.calculateGradients(null, "x");
        INDArray grad = gradMap.get("x");

        assertNotNull(grad, "Gradient should not be null");
        assertEquals(input.shape(), grad.shape(), "Gradient shape should match input");
    }

    /**
     * Test Relu6 gradient backprop.
     */
    @Test
    public void testRelu6Bp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.create(new float[]{-1, 0, 3, 7}, new int[]{2, 2});

        var x = sd.var("x", input);
        var relu6 = sd.nn().relu6(x, 0);
        var loss = relu6.sum();

        sd.setLossVariables(loss);
        sd.createGradFunction();

        var gradMap = sd.calculateGradients(null, "x");
        INDArray grad = gradMap.get("x");

        assertNotNull(grad, "Gradient should not be null");
        // Gradient should be 0 where x < 0 or x > 6, 1 otherwise
        assertEquals(0.0, grad.getDouble(0), 1e-5, "Gradient should be 0 for negative input");
        assertEquals(1.0, grad.getDouble(2), 1e-5, "Gradient should be 1 for input in [0, 6]");
        assertEquals(0.0, grad.getDouble(3), 1e-5, "Gradient should be 0 for input > 6");
    }

    /**
     * Test LeakyReLU gradient backprop.
     */
    @Test
    public void testLeakyReluBp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.create(new float[]{-1, 0, 1, 2}, new int[]{2, 2});

        var x = sd.var("x", input);
        var leakyRelu = sd.nn().leakyRelu(x, 0.01);
        var loss = leakyRelu.sum();

        sd.setLossVariables(loss);
        sd.createGradFunction();

        var gradMap = sd.calculateGradients(null, "x");
        INDArray grad = gradMap.get("x");

        assertNotNull(grad, "Gradient should not be null");
        assertEquals(0.01, grad.getDouble(0), 1e-5, "Gradient should be alpha for negative input");
        assertEquals(1.0, grad.getDouble(2), 1e-5, "Gradient should be 1 for positive input");
    }

    /**
     * Test Softplus gradient backprop.
     */
    @Test
    public void testSoftplusBp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.create(new float[]{-1, 0, 1, 2}, new int[]{2, 2});

        var x = sd.var("x", input);
        var softplus = sd.nn().softplus(x);
        var loss = softplus.sum();

        sd.setLossVariables(loss);
        sd.createGradFunction();

        var gradMap = sd.calculateGradients(null, "x");
        INDArray grad = gradMap.get("x");

        // Softplus gradient: sigmoid(x)
        assertNotNull(grad, "Gradient should not be null");
        assertEquals(input.shape(), grad.shape(), "Gradient shape should match input");
    }

    /**
     * Test Softsign gradient backprop.
     */
    @Test
    public void testSoftsignBp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.create(new float[]{-1, 0, 1, 2}, new int[]{2, 2});

        var x = sd.var("x", input);
        var softsign = sd.nn().softsign(x);
        var loss = softsign.sum();

        sd.setLossVariables(loss);
        sd.createGradFunction();

        var gradMap = sd.calculateGradients(null, "x");
        INDArray grad = gradMap.get("x");

        assertNotNull(grad, "Gradient should not be null");
        assertEquals(input.shape(), grad.shape(), "Gradient shape should match input");
    }

    /**
     * Test reverse op (reverse along axis).
     */
    @Test
    public void testReverseOp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});

        var x = sd.var("x", input);
        // Reverse along axis 1
        var result = sd.reverse(x, 1);

        INDArray output = result.eval();
        INDArray expected = Nd4j.create(new float[]{3, 2, 1, 6, 5, 4}, new int[]{2, 3});
        assertEquals(expected, output, "Reverse along axis 1 should reverse columns");
    }

    /**
     * Test MatrixBandPart op.
     */
    @Test
    public void testMatrixBandPartOp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.ones(DataType.FLOAT, 3, 3);

        var x = sd.var("x", input);
        // Keep main diagonal only (numLower=0, numUpper=0)
        var result = sd.linalg().matrixBandPart(x, 0, 0);

        // matrixBandPart returns an array
        assertNotNull(result, "MatrixBandPart should return result");
        assertTrue(result.length > 0, "Result should have at least one output");
    }

    /**
     * Test QR decomposition op.
     */
    @Test
    public void testQrOp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, new int[]{3, 3});

        var x = sd.var("x", input);
        var qr = sd.linalg().qr(x);

        // QR returns [Q, R] array
        assertNotNull(qr, "QR decomposition should return result");
        assertEquals(2, qr.length, "QR should return 2 matrices (Q and R)");
    }

    /**
     * Test Softmax gradient backprop.
     */
    @Test
    public void testSoftmaxBp() {
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});

        var x = sd.var("x", input);
        var softmax = sd.nn().softmax(x, 1);
        var loss = softmax.sum();

        sd.setLossVariables(loss);
        sd.createGradFunction();

        var gradMap = sd.calculateGradients(null, "x");
        INDArray grad = gradMap.get("x");

        assertNotNull(grad, "Gradient should not be null");
        assertEquals(input.shape(), grad.shape(), "Gradient shape should match input");
    }

    /**
     * Test that mapping processes can be retrieved and have correct structure.
     */
    @Test
    public void testMappingProcessStructure() {
        var registry = TensorflowOpDeclarationsKt.registry();

        // Test Snapshot mapping
        var snapshotProcess = registry.lookupOpMappingProcess("Snapshot");
        assertNotNull(snapshotProcess, "Snapshot process should exist");
        assertEquals("identity", snapshotProcess.opName(), "Snapshot should map to identity");

        // Test PreventGradient mapping
        var preventGradProcess = registry.lookupOpMappingProcess("PreventGradient");
        assertNotNull(preventGradProcess, "PreventGradient process should exist");
        assertEquals("stop_gradient", preventGradProcess.opName(), "PreventGradient should map to stop_gradient");

        // Test Copy mapping
        var copyProcess = registry.lookupOpMappingProcess("Copy");
        assertNotNull(copyProcess, "Copy process should exist");
        assertEquals("copy", copyProcess.opName(), "Copy should map to copy op");

        // Test Einsum mapping
        var einsumProcess = registry.lookupOpMappingProcess("Einsum");
        assertNotNull(einsumProcess, "Einsum process should exist");
        assertEquals("einsum", einsumProcess.opName(), "Einsum should map to einsum op");

        // Test Qr mapping
        var qrProcess = registry.lookupOpMappingProcess("Qr");
        assertNotNull(qrProcess, "Qr process should exist");
        assertEquals("qr", qrProcess.opName(), "Qr should map to qr op");

        // Test Eig mapping
        var eigProcess = registry.lookupOpMappingProcess("Eig");
        assertNotNull(eigProcess, "Eig process should exist");
        assertEquals("eig", eigProcess.opName(), "Eig should map to eig op");
    }

    /**
     * Test gradient op mapping targets.
     */
    @Test
    public void testGradientOpMappingTargets() {
        var registry = TensorflowOpDeclarationsKt.registry();

        // Test that gradient ops map to their _bp counterparts
        assertEquals("sigmoid_bp", registry.lookupOpMappingProcess("SigmoidGrad").opName(),
                "SigmoidGrad should map to sigmoid_bp");
        assertEquals("tanh_bp", registry.lookupOpMappingProcess("TanhGrad").opName(),
                "TanhGrad should map to tanh_bp");
        assertEquals("relu_bp", registry.lookupOpMappingProcess("ReluGrad").opName(),
                "ReluGrad should map to relu_bp");
        assertEquals("elu_bp", registry.lookupOpMappingProcess("EluGrad").opName(),
                "EluGrad should map to elu_bp");
        assertEquals("selu_bp", registry.lookupOpMappingProcess("SeluGrad").opName(),
                "SeluGrad should map to selu_bp");
        assertEquals("relu6_bp", registry.lookupOpMappingProcess("Relu6Grad").opName(),
                "Relu6Grad should map to relu6_bp");
        assertEquals("softplus_bp", registry.lookupOpMappingProcess("SoftplusGrad").opName(),
                "SoftplusGrad should map to softplus_bp");
        assertEquals("softsign_bp", registry.lookupOpMappingProcess("SoftsignGrad").opName(),
                "SoftsignGrad should map to softsign_bp");
        assertEquals("lrelu_bp", registry.lookupOpMappingProcess("LeakyReluGrad").opName(),
                "LeakyReluGrad should map to lrelu_bp");
    }

    /**
     * Test safe math op mapping targets.
     */
    @Test
    public void testSafeMathOpMappingTargets() {
        var registry = TensorflowOpDeclarationsKt.registry();

        assertEquals("xdivy", registry.lookupOpMappingProcess("Xdivy").opName(),
                "Xdivy should map to xdivy");
        assertEquals("xlogy", registry.lookupOpMappingProcess("Xlogy").opName(),
                "Xlogy should map to xlogy");
        assertEquals("xlog1py", registry.lookupOpMappingProcess("Xlog1py").opName(),
                "Xlog1py should map to xlog1py");
    }

    /**
     * Test array op mapping targets.
     */
    @Test
    public void testArrayOpMappingTargets() {
        var registry = TensorflowOpDeclarationsKt.registry();

        assertEquals("reverse", registry.lookupOpMappingProcess("ReverseV2").opName(),
                "ReverseV2 should map to reverse");
        assertEquals("matrix_band_part", registry.lookupOpMappingProcess("MatrixBandPart").opName(),
                "MatrixBandPart should map to matrix_band_part");
    }
}
