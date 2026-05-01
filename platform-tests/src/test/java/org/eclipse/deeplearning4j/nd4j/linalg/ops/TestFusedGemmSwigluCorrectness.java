package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedGemmSwiglu;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness test for fused_gemm_swiglu: verifies that the fused op
 * produces the same result as the unfused equivalent, including after
 * SameDiff.dup() round-trip.
 */
public class TestFusedGemmSwigluCorrectness {

    @Test
    public void testFusedMatchesUnfused_rank2() {
        int M = 4, K = 64, N = 32;
        INDArray x = Nd4j.rand(DataType.FLOAT, M, K);
        INDArray wGate = Nd4j.rand(DataType.FLOAT, K, N);
        INDArray wUp = Nd4j.rand(DataType.FLOAT, K, N);

        // Unfused reference: silu(x @ wGate) * (x @ wUp)
        INDArray gate = x.mmul(wGate);           // [M, N]
        INDArray up = x.mmul(wUp);               // [M, N]
        INDArray silu = Nd4j.nn().silu(gate);     // silu(gate)
        INDArray expected = silu.mul(up);          // silu(gate) * up

        // === Direct INDArray op (bypasses SameDiff/DSP) ===
        INDArray directResult = Nd4j.create(DataType.FLOAT, M, N);
        FusedGemmSwiglu directOp = new FusedGemmSwiglu(x, wGate, wUp, directResult);
        Nd4j.exec(directOp);

        System.out.println("Expected first row:  " + expected.getRow(0));
        System.out.println("Direct first row:    " + directResult.getRow(0));
        double directMaxDiff = expected.sub(directResult).amaxNumber().doubleValue();
        System.out.println("Direct maxDiff: " + directMaxDiff);

        // === SameDiff op (goes through DSP) ===
        SameDiff sd = SameDiff.create();
        SDVariable xVar = sd.var("x", x.dup());
        SDVariable wGateVar = sd.var("wGate", wGate.dup());
        SDVariable wUpVar = sd.var("wUp", wUp.dup());
        SDVariable out = new FusedGemmSwiglu(sd, xVar, wGateVar, wUpVar).outputVariable();

        Map<String, INDArray> results = sd.outputAll(null);
        INDArray actual = results.get(out.name());

        assertArrayEquals(new long[]{M, N}, actual.shape());

        System.out.println("SameDiff first row:  " + actual.getRow(0));
        double sdMaxDiff = expected.sub(actual).amaxNumber().doubleValue();
        System.out.println("SameDiff maxDiff: " + sdMaxDiff);

        // Direct op must match reference
        assertTrue(expected.equalsWithEps(directResult, 1e-2),
                "Direct fused op should match unfused reference, maxDiff=" + directMaxDiff);

        // SameDiff op must match reference
        assertTrue(expected.equalsWithEps(actual, 1e-2),
                "SameDiff fused op should match unfused reference (rank 2), maxDiff=" + sdMaxDiff);
    }

    @Test
    public void testFusedMatchesUnfused_rank3() {
        // Rank 3: typical LLM decode shape [batch, seq, hidden]
        int B = 1, S = 1, K = 64, N = 32;
        INDArray x = Nd4j.rand(DataType.FLOAT, B, S, K);
        INDArray wGate = Nd4j.rand(DataType.FLOAT, K, N);
        INDArray wUp = Nd4j.rand(DataType.FLOAT, K, N);

        // Unfused reference
        INDArray xFlat = x.reshape(B * S, K);
        INDArray gate = xFlat.mmul(wGate);
        INDArray up = xFlat.mmul(wUp);
        INDArray silu = Nd4j.nn().silu(gate);
        INDArray expectedFlat = silu.mul(up);
        INDArray expected = expectedFlat.reshape(B, S, N);

        // Fused op
        SameDiff sd = SameDiff.create();
        SDVariable xVar = sd.var("x", x.dup());
        SDVariable wGateVar = sd.var("wGate", wGate.dup());
        SDVariable wUpVar = sd.var("wUp", wUp.dup());
        SDVariable out = new FusedGemmSwiglu(sd, xVar, wGateVar, wUpVar).outputVariable();

        Map<String, INDArray> results = sd.outputAll(null);
        INDArray actual = results.get(out.name());

        assertArrayEquals(new long[]{B, S, N}, actual.shape());
        assertTrue(expected.equalsWithEps(actual, 1e-1),
                "Fused op should match unfused reference (rank 3), maxDiff=" + expected.sub(actual).amaxNumber());
    }

    @Test
    public void testFusedMatchesUnfused_afterDup() {
        // Test that SameDiff.dup() round-trip preserves input ordering
        int M = 2, K = 32, N = 16;
        INDArray x = Nd4j.rand(DataType.FLOAT, M, K);
        INDArray wGate = Nd4j.rand(DataType.FLOAT, K, N);
        INDArray wUp = Nd4j.rand(DataType.FLOAT, K, N);

        // Unfused reference
        INDArray gate = x.mmul(wGate);
        INDArray up = x.mmul(wUp);
        INDArray silu = Nd4j.nn().silu(gate);
        INDArray expected = silu.mul(up);

        // Build graph, dup it, then run
        SameDiff sd = SameDiff.create();
        SDVariable xVar = sd.var("x", x.dup());
        SDVariable wGateVar = sd.var("wGate", wGate.dup());
        SDVariable wUpVar = sd.var("wUp", wUp.dup());
        SDVariable out = new FusedGemmSwiglu(sd, xVar, wGateVar, wUpVar).outputVariable();
        String outName = out.name();

        // dup() round-trip (FlatBuffers serialize → deserialize)
        SameDiff duped = sd.dup();
        Map<String, INDArray> results = duped.outputAll(null);
        INDArray actual = results.get(outName);

        assertNotNull(actual, "Output should exist after dup()");
        assertArrayEquals(new long[]{M, N}, actual.shape());
        assertTrue(expected.equalsWithEps(actual, 1e-2),
                "Fused op should match unfused reference after dup(), maxDiff=" + expected.sub(actual).amaxNumber());
    }

    @Test
    public void testGraphOptimizerFusion() {
        // Build the unfused graph: sigmoid(mmul(x, wGate)) * mmul(x, wGate) * mmul(x, wUp)
        // Run GraphOptimizer — should fuse sigmoid*x → swish, then swish*y → swish_mul
        // (FuseGatedMLPPattern is disabled because fused_gemm_swiglu regresses DSP throughput)
        int M = 2, K = 32, N = 16;
        INDArray xData = Nd4j.rand(DataType.FLOAT, M, K);
        INDArray wGateData = Nd4j.rand(DataType.FLOAT, K, N);
        INDArray wUpData = Nd4j.rand(DataType.FLOAT, K, N);

        // Build unfused graph
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.var("x", xData.dup());
        SDVariable wGate = sd.var("wGate", wGateData.dup());
        SDVariable wUp = sd.var("wUp", wUpData.dup());

        // matmul(x, wGate) → sigmoid → mul with matmul output → swish
        // matmul(x, wUp) → multiplied with swish result
        SDVariable gateOut = sd.linalg().matmul("gate_matmul", x, wGate);
        SDVariable upOut = sd.linalg().matmul("up_matmul", x, wUp);
        SDVariable sigmoid = sd.nn().sigmoid("sigmoid", gateOut);
        SDVariable swish = sigmoid.mul("swish", gateOut);   // sigmoid(gate) * gate = silu(gate)
        SDVariable output = swish.mul("output", upOut);      // silu(gate) * up

        // Run unfused
        Map<String, INDArray> unfusedResults = sd.output((Map<String, INDArray>) null, "output");
        INDArray unfusedOutput = unfusedResults.get("output");

        // Run GraphOptimizer — should fuse to swish_mul (not fused_gemm_swiglu)
        SameDiff optimized = org.nd4j.autodiff.samediff.optimize.GraphOptimizer.optimize(
                sd, java.util.Collections.singletonList("output"));

        // Check that swish_mul was created (sigmoid*x→swish then swish*y→swish_mul)
        boolean hasSwishMul = optimized.getOps().values().stream()
                .anyMatch(op -> op.getOp() instanceof org.nd4j.linalg.api.ops.impl.transforms.custom.SwishMul);
        System.out.println("Has swish_mul: " + hasSwishMul);
        System.out.println("Ops after optimization: " + optimized.getOps().size());

        // Run optimized
        Map<String, INDArray> fusedResults = optimized.output((Map<String, INDArray>) null, "output");
        INDArray fusedOutput = fusedResults.get("output");

        assertNotNull(fusedOutput);
        assertArrayEquals(unfusedOutput.shape(), fusedOutput.shape());

        double maxDiff = unfusedOutput.sub(fusedOutput).amaxNumber().doubleValue();
        System.out.println("Max diff between optimized and unfused: " + maxDiff);
        System.out.println("Unfused first row: " + unfusedOutput.getRow(0));
        System.out.println("Optimized first row:   " + fusedOutput.getRow(0));

        assertTrue(unfusedOutput.equalsWithEps(fusedOutput, 1e-1),
                "GraphOptimizer-fused result should match unfused, maxDiff=" + maxDiff);
    }

    @Test
    public void testSwappedWeightsProduceDifferentOutput() {
        // Verify that swapping wGate and wUp produces DIFFERENT output
        // (to confirm the op is not commutative and order matters)
        int M = 2, K = 32, N = 16;
        INDArray x = Nd4j.rand(DataType.FLOAT, M, K);
        INDArray w1 = Nd4j.rand(DataType.FLOAT, K, N);
        INDArray w2 = Nd4j.rand(DataType.FLOAT, K, N);

        // Forward order: silu(x @ w1) * (x @ w2)
        SameDiff sd1 = SameDiff.create();
        SDVariable out1 = new FusedGemmSwiglu(sd1,
                sd1.var("x", x.dup()), sd1.var("w1", w1.dup()), sd1.var("w2", w2.dup())).outputVariable();
        INDArray result1 = sd1.outputAll(null).get(out1.name());

        // Swapped order: silu(x @ w2) * (x @ w1)
        SameDiff sd2 = SameDiff.create();
        SDVariable out2 = new FusedGemmSwiglu(sd2,
                sd2.var("x", x.dup()), sd2.var("w1", w2.dup()), sd2.var("w2", w1.dup())).outputVariable();
        INDArray result2 = sd2.outputAll(null).get(out2.name());

        assertNotEquals(result1, result2, "Swapping weights should produce different results");
    }
}
