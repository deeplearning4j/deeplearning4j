package org.eclipse.deeplearning4j.nd4j.autodiff.optimization;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Diagnostic tests to isolate optimizer correctness issues.
 */
public class TestOptimizerDiag {

    /**
     * tanh -> abs -> neg chain WITHOUT optimizer.
     * Test each op individually to find where the divergence starts.
     */
    @Test
    void testUnaryChain_NoOptimizer() {
        INDArray input = Nd4j.create(new float[]{-2f, -1f, 0f, 1f, 2f}, new long[]{1, 5});

        // Step 1: tanh alone
        SameDiff sd1 = SameDiff.create();
        SDVariable x1 = sd1.placeHolder("x", DataType.FLOAT, -1, -1);
        sd1.math().tanh("output", x1);
        sd1.setOutputs("output");
        INDArray tanhSd = sd1.output(Map.of("x", input), "output").get("output");
        INDArray tanhNd4j = Nd4j.math().tanh(input.dup());
        double tanhDiff = tanhSd.sub(tanhNd4j).amaxNumber().doubleValue();

        // Step 2: abs alone
        SameDiff sd2 = SameDiff.create();
        SDVariable x2 = sd2.placeHolder("x", DataType.FLOAT, -1, -1);
        sd2.math().abs("output", x2);
        sd2.setOutputs("output");
        INDArray absInput = tanhNd4j.dup();
        INDArray absSd = sd2.output(Map.of("x", absInput), "output").get("output");
        INDArray absNd4j = Nd4j.math().abs(absInput.dup());
        double absDiff = absSd.sub(absNd4j).amaxNumber().doubleValue();

        // Step 3: neg alone
        SameDiff sd3 = SameDiff.create();
        SDVariable x3 = sd3.placeHolder("x", DataType.FLOAT, -1, -1);
        sd3.math().neg("output", x3);
        sd3.setOutputs("output");
        INDArray negInput = absNd4j.dup();
        INDArray negSd = sd3.output(Map.of("x", negInput), "output").get("output");
        INDArray negNd4j = negInput.dup().neg();
        double negDiff = negSd.sub(negNd4j).amaxNumber().doubleValue();

        // Step 4: full chain
        SameDiff sdFull = SameDiff.create();
        SDVariable xf = sdFull.placeHolder("x", DataType.FLOAT, -1, -1);
        sdFull.math().tanh("tanh_out", xf);
        sdFull.math().abs("abs_out", sdFull.getVariable("tanh_out"));
        sdFull.math().neg("output", sdFull.getVariable("abs_out"));
        sdFull.setOutputs("output");
        INDArray fullResult = sdFull.output(Map.of("x", input), "output").get("output");
        INDArray fullExpected = Nd4j.math().tanh(input.dup());
        fullExpected = Nd4j.math().abs(fullExpected.dup());
        fullExpected = fullExpected.neg();
        double fullDiff = fullResult.sub(fullExpected).amaxNumber().doubleValue();

        // Also request intermediate outputs
        SameDiff sdDebug = SameDiff.create();
        SDVariable xd = sdDebug.placeHolder("x", DataType.FLOAT, -1, -1);
        sdDebug.math().tanh("tanh_out", xd);
        sdDebug.math().abs("abs_out", sdDebug.getVariable("tanh_out"));
        sdDebug.math().neg("output", sdDebug.getVariable("abs_out"));
        sdDebug.setOutputs("tanh_out", "abs_out", "output");
        Map<String, INDArray> debugOutputs = sdDebug.output(Map.of("x", input), "tanh_out", "abs_out", "output");

        System.err.println("=== Per-op diagnostics ===");
        System.err.println("tanh diff: " + tanhDiff);
        System.err.println("abs diff:  " + absDiff);
        System.err.println("neg diff:  " + negDiff);
        System.err.println("full chain diff: " + fullDiff);
        System.err.println("--- Chain intermediates ---");
        System.err.println("input:          " + input);
        System.err.println("tanh(SameDiff): " + debugOutputs.get("tanh_out"));
        System.err.println("tanh(Nd4j):     " + tanhNd4j);
        System.err.println("abs(SameDiff):  " + debugOutputs.get("abs_out"));
        System.err.println("output(SD):     " + debugOutputs.get("output"));
        System.err.println("expected:       " + fullExpected);

        assertTrue(tanhDiff < 1e-5, "tanh diverges: " + tanhDiff);
        assertTrue(absDiff < 1e-5, "abs diverges: " + absDiff);
        assertTrue(negDiff < 1e-5, "neg diverges: " + negDiff);
        assertTrue(fullDiff < 1e-5, "full chain diverges: " + fullDiff);
    }

    /**
     * tanh -> abs -> neg chain WITH optimizer.
     * If the no-optimizer test passes but this one fails, the optimizer is breaking the graph.
     */
    @Test
    void testUnaryChain_WithOptimizer() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1);
        SDVariable t = sd.math().tanh("tanh_out", x);
        SDVariable a = sd.math().abs("abs_out", t);
        SDVariable n = sd.math().neg("output", a);
        sd.setOutputs("output");

        INDArray input = Nd4j.create(new float[]{-2f, -1f, 0f, 1f, 2f}, new long[]{1, 5});
        INDArray expected = Nd4j.math().tanh(input.dup());
        expected = Nd4j.math().abs(expected.dup());
        expected = expected.neg();

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        // Print the optimized graph ops
        System.out.println("=== Optimized graph ops ===");
        for (Map.Entry<String, SameDiffOp> e : optimized.getOps().entrySet()) {
            SameDiffOp op = e.getValue();
            System.out.println("  " + e.getKey() + " [" + op.getOp().opName() + "] inputs="
                + op.getInputsToOp() + " outputs=" + op.getOutputsOfOp());
        }
        System.out.println("Graph outputs: " + optimized.outputs());

        INDArray result = optimized.output(Map.of("x", input), "output").get("output");
        System.out.println("Expected: " + expected);
        System.out.println("Got:      " + result);
        double maxDiff = expected.sub(result).amaxNumber().doubleValue();
        System.out.println("Max diff (with optimizer): " + maxDiff);

        assertTrue(maxDiff < 1e-5, "Optimizer broke the graph, diff = " + maxDiff);
    }

    /**
     * Simple tanh -> sigmoid (multi-step from buffer aliasing test).
     */
    @Test
    void testTanhSigmoid_NoOptimizer() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable t = sd.math().tanh("t", x);
        SDVariable s = sd.nn().sigmoid("output", t);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 16);
        INDArray expected = Nd4j.math().tanh(input.dup());
        expected = Nd4j.nn().sigmoid(expected.dup());

        INDArray result = sd.output(Map.of("x", input), "output").get("output");
        double maxDiff = expected.sub(result).amaxNumber().doubleValue();
        System.out.println("tanh->sigmoid diff (no optimizer): " + maxDiff);
        assertTrue(maxDiff < 1e-5, "tanh->sigmoid no-opt diff = " + maxDiff);
    }

    @Test
    void testTanhSigmoid_WithOptimizer() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable t = sd.math().tanh("t", x);
        SDVariable s = sd.nn().sigmoid("output", t);
        sd.setOutputs("output");

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 16);
        INDArray expected = Nd4j.math().tanh(input.dup());
        expected = Nd4j.nn().sigmoid(expected.dup());

        SameDiff optimized = GraphOptimizer.optimize(sd, "output");

        System.out.println("=== Optimized graph ops ===");
        for (Map.Entry<String, SameDiffOp> e : optimized.getOps().entrySet()) {
            SameDiffOp op = e.getValue();
            System.out.println("  " + e.getKey() + " [" + op.getOp().opName() + "] inputs="
                + op.getInputsToOp() + " outputs=" + op.getOutputsOfOp());
        }

        INDArray result = optimized.output(Map.of("x", input), "output").get("output");
        double maxDiff = expected.sub(result).amaxNumber().doubleValue();
        System.out.println("tanh->sigmoid diff (with optimizer): " + maxDiff);
        assertTrue(maxDiff < 1e-5, "Optimizer broke tanh->sigmoid, diff = " + maxDiff);
    }

    /**
     * Horizontal fusion correctness - 2 parallel matmuls.
     */
    @Test
    void testParallelMatmul_WithOptimizer() {
        SameDiff sd = SameDiff.create();
        int d = 8, h1 = 4, h2 = 4;
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, d);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, d, h1));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, d, h2));
        sd.mmul("out1", x, w1);
        sd.mmul("out2", x, w2);
        sd.setOutputs("out1", "out2");

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, d);

        // Reference: no optimizer
        INDArray refOut1 = sd.output(Map.of("x", input), "out1", "out2").get("out1").dup();
        INDArray refOut2 = sd.output(Map.of("x", input), "out1", "out2").get("out2").dup();

        SameDiff optimized = GraphOptimizer.optimize(sd, "out1", "out2");

        System.out.println("=== Optimized graph ops ===");
        for (Map.Entry<String, SameDiffOp> e : optimized.getOps().entrySet()) {
            SameDiffOp op = e.getValue();
            System.out.println("  " + e.getKey() + " [" + op.getOp().opName() + "] inputs="
                + op.getInputsToOp() + " outputs=" + op.getOutputsOfOp());
        }

        Map<String, INDArray> optOutputs = optimized.output(Map.of("x", input), "out1", "out2");
        double diff1 = refOut1.sub(optOutputs.get("out1")).amaxNumber().doubleValue();
        double diff2 = refOut2.sub(optOutputs.get("out2")).amaxNumber().doubleValue();
        System.out.println("out1 diff: " + diff1);
        System.out.println("out2 diff: " + diff2);

        assertTrue(diff1 < 1e-5, "Horizontal fusion broke out1, diff = " + diff1);
        assertTrue(diff2 < 1e-5, "Horizontal fusion broke out2, diff = " + diff2);
    }
}
