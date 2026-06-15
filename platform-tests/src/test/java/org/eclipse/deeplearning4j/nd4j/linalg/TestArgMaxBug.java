package org.eclipse.deeplearning4j.nd4j.linalg;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Standalone test for Nd4j.argMax bug on large arrays (e.g., vocab-sized 49280).
 *
 * The CUDA IndexReduce scalar kernel uses multi-block reduction for arrays
 * larger than 256 elements. A previous fix (c23c6104fd3) stored IndexValue
 * structs (value + index) per block, but argMax may still return wrong indices
 * when the true maximum is in a high-index block.
 */
public class TestArgMaxBug {

    @Test
    public void testArgMaxScalar_HighIndex() {
        // Simulate VLM logits: 49280 elements, max at high index
        int vocabSize = 49280;
        int expectedMaxIdx = 49229;  // like <doctag> token

        INDArray logits = Nd4j.randn(DataType.FLOAT, vocabSize).muli(2.0);
        // Set a clear maximum at the high index
        logits.putScalar(expectedMaxIdx, 100.0f);

        INDArray result = Nd4j.argMax(logits);
        int actualIdx = result.getInt(0);

        System.out.println("Expected argmax: " + expectedMaxIdx + ", Got: " + actualIdx);
        System.out.println("Value at expected: " + logits.getFloat(expectedMaxIdx));
        System.out.println("Value at actual: " + logits.getFloat(actualIdx));

        assertEquals(expectedMaxIdx, actualIdx,
                "argMax should return " + expectedMaxIdx + " but got " + actualIdx);
    }

    @Test
    public void testArgMaxScalar_VariousPositions() {
        // Test argmax at various positions across the array
        int vocabSize = 49280;
        int[] testPositions = {0, 1, 127, 255, 256, 511, 1000, 10000, 25000, 49000, 49229, 49279};

        for (int expectedIdx : testPositions) {
            INDArray logits = Nd4j.zeros(DataType.FLOAT, vocabSize);
            logits.putScalar(expectedIdx, 10.0f);

            int actualIdx = Nd4j.argMax(logits).getInt(0);

            System.out.println("Position " + expectedIdx + ": argMax=" + actualIdx +
                    " " + (actualIdx == expectedIdx ? "OK" : "FAIL"));

            assertEquals(expectedIdx, actualIdx,
                    "argMax failed at position " + expectedIdx + ": got " + actualIdx);
        }
    }

    @Test
    public void testArgMaxScalar_RealisticLogits() {
        // Create realistic logit distribution: mostly small values, one clear winner
        int vocabSize = 49280;
        int expectedMaxIdx = 49229;

        // Fill with values similar to real decoder logits
        INDArray logits = Nd4j.randn(DataType.FLOAT, vocabSize);  // ~N(0,1)
        // Set a clear winner at the target position (like real decoder output)
        logits.putScalar(expectedMaxIdx, 13.27f);  // actual value from batch test
        // Set a runner-up (like <chart> token at 49247)
        logits.putScalar(49247, 3.69f);

        int actualIdx = Nd4j.argMax(logits).getInt(0);

        System.out.println("Realistic logits: expected=" + expectedMaxIdx + ", got=" + actualIdx);
        System.out.println("Max value: " + logits.getFloat(expectedMaxIdx));

        assertEquals(expectedMaxIdx, actualIdx,
                "argMax should return " + expectedMaxIdx + " but got " + actualIdx);
    }

    @Test
    public void testArgMaxDim1_2D() {
        // Test the 2D argMax path used by argmaxBatch
        int vocabSize = 49280;
        int batchSize = 1;
        int expectedMaxIdx = 49229;

        INDArray logits = Nd4j.randn(DataType.FLOAT, batchSize, vocabSize);
        logits.putScalar(new int[]{0, expectedMaxIdx}, 100.0f);

        INDArray result = Nd4j.argMax(logits, 1);
        int actualIdx = result.getInt(0);

        System.out.println("2D argMax(dim=1): expected=" + expectedMaxIdx + ", got=" + actualIdx);

        assertEquals(expectedMaxIdx, actualIdx,
                "2D argMax should return " + expectedMaxIdx + " but got " + actualIdx);
    }

    @Test
    public void testArgMaxConsistency_MultipleRuns() {
        // Run argMax multiple times on the same data to check for non-determinism
        int vocabSize = 49280;
        int expectedMaxIdx = 49229;

        INDArray logits = Nd4j.zeros(DataType.FLOAT, vocabSize);
        logits.putScalar(expectedMaxIdx, 100.0f);
        logits.putScalar(1, 5.0f);  // block 0 has a local max at index 1

        int failures = 0;
        for (int i = 0; i < 100; i++) {
            int actualIdx = Nd4j.argMax(logits.dup()).getInt(0);
            if (actualIdx != expectedMaxIdx) {
                failures++;
                if (failures <= 5) {
                    System.out.println("Run " + i + ": expected " + expectedMaxIdx + " got " + actualIdx);
                }
            }
        }

        System.out.println("Consistency: " + (100 - failures) + "/100 correct");
        assertEquals(0, failures, failures + " out of 100 runs returned wrong argmax");
    }

    @Test
    public void testArgMax_AfterHeavyGpuWork() {
        // Simulate the VLM context: heavy GPU work (matmul, etc.) then argMax
        int vocabSize = 49280;
        int expectedMaxIdx = 49229;

        // Do some heavy GPU work first (like DSP execution)
        for (int i = 0; i < 5; i++) {
            INDArray a = Nd4j.randn(DataType.FLOAT, 512, 576);
            INDArray b = Nd4j.randn(DataType.FLOAT, 576, 49280);
            INDArray c = a.mmul(b);  // matmul like decoder
            c.close();
            a.close();
            b.close();
        }

        // Now test argMax — this simulates the logits argMax after decoder execution
        INDArray logits = Nd4j.randn(DataType.FLOAT, vocabSize);
        logits.putScalar(expectedMaxIdx, 100.0f);

        // Don't explicitly commit/sync — test if argMax works without it
        int actualIdx = Nd4j.argMax(logits).getInt(0);

        System.out.println("After heavy GPU work: expected=" + expectedMaxIdx + ", got=" + actualIdx);
        assertEquals(expectedMaxIdx, actualIdx,
                "argMax after heavy GPU work should return " + expectedMaxIdx + " but got " + actualIdx);
    }

    @Test
    public void testArgMax_AfterSameDiffExecution() {
        // Test argMax after running a SameDiff graph — this is closest to the VLM context
        int vocabSize = 49280;
        int expectedMaxIdx = 49229;

        // Build a simple SameDiff graph (like embed_tokens)
        org.nd4j.autodiff.samediff.SameDiff sd = org.nd4j.autodiff.samediff.SameDiff.create();
        org.nd4j.autodiff.samediff.SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 576);
        org.nd4j.autodiff.samediff.SDVariable weight = sd.constant("weight", Nd4j.randn(DataType.FLOAT, 576, 49280));
        org.nd4j.autodiff.samediff.SDVariable output = sd.mmul("output", input, weight);

        // Execute the graph a few times (like repeated decoder calls)
        for (int i = 0; i < 3; i++) {
            INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, 576);
            java.util.Map<String, INDArray> results = sd.output(
                    java.util.Map.of("input", inputArr), "output");
            INDArray result = results.get("output");

            // Now immediately try argMax on a vocab-sized array — does it work?
            INDArray logits = Nd4j.randn(DataType.FLOAT, vocabSize);
            logits.putScalar(expectedMaxIdx, 100.0f);

            int actualIdx = Nd4j.argMax(logits).getInt(0);
            System.out.println("After SameDiff exec " + i + ": expected=" + expectedMaxIdx + ", got=" + actualIdx);

            assertEquals(expectedMaxIdx, actualIdx,
                    "argMax after SameDiff exec should return " + expectedMaxIdx + " but got " + actualIdx);

            logits.close();
        }
    }

    @Test
    public void testArgMax_OnSameDiffOutput() {
        // Test argMax directly on SameDiff output — closest to the actual VLM code path
        int vocabSize = 49280;
        int hiddenSize = 576;

        // Build a graph that produces [1, 1, vocabSize] output (like decoder logits)
        org.nd4j.autodiff.samediff.SameDiff sd = org.nd4j.autodiff.samediff.SameDiff.create();
        org.nd4j.autodiff.samediff.SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, hiddenSize);
        org.nd4j.autodiff.samediff.SDVariable weight = sd.constant("weight", Nd4j.randn(DataType.FLOAT, hiddenSize, vocabSize));
        org.nd4j.autodiff.samediff.SDVariable output = sd.mmul("logits", input, weight);

        // Execute graph
        INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, hiddenSize);
        java.util.Map<String, INDArray> results = sd.output(
                java.util.Map.of("input", inputArr), "logits");
        INDArray logitsRaw = results.get("logits");  // [1, vocabSize]

        // Extract and dup like VLM code
        INDArray logits = logitsRaw.dup();

        // Get argmax two ways — they should agree
        INDArray flat = logits.reshape(logits.length());
        int argmaxResult = Nd4j.argMax(flat).getInt(0);

        // Also get it via sortWithIndices
        INDArray[] sortResult = Nd4j.sortWithIndices(flat.dup(), 0, false);
        int sortArgmax = sortResult[0].getInt(0);

        System.out.println("SameDiff output argmax: argMax=" + argmaxResult + ", sort=" + sortArgmax);
        System.out.println("Value at argMax result: " + flat.getFloat(argmaxResult));
        System.out.println("Value at sort result: " + flat.getFloat(sortArgmax));

        assertEquals(sortArgmax, argmaxResult,
                "argMax (" + argmaxResult + ") and sortWithIndices (" + sortArgmax + ") should agree");
    }

    @Test
    public void testToIntVector_VsGetInt_AfterGpuWork() {
        // Test whether toIntVector() returns the same result as getInt(0)
        // on a CUDA argMax result. toIntVector() uses data().asInt() which
        // calls getIntUnsynced() — this bypasses CUDA device-to-host sync.
        // getInt(0) calls synchronizeHostData first.
        int vocabSize = 49280;
        int expectedMaxIdx = 49229;

        // Do heavy GPU work to build up async pipeline state
        for (int i = 0; i < 5; i++) {
            INDArray a = Nd4j.randn(DataType.FLOAT, 512, 576);
            INDArray b = Nd4j.randn(DataType.FLOAT, 576, vocabSize);
            INDArray c = a.mmul(b);
            c.close();
            a.close();
            b.close();
        }

        // Create 2D logits [1, vocabSize] to go through argMax(dim=1) path
        INDArray logits = Nd4j.randn(DataType.FLOAT, 1, vocabSize);
        logits.putScalar(new int[]{0, expectedMaxIdx}, 100.0f);

        // argMax along dim 1 — returns shape [1]
        INDArray argMaxResult = Nd4j.argMax(logits, 1);

        // Read via getInt (WITH sync)
        int viaGetInt = argMaxResult.getInt(0);

        // Read via toIntVector (may SKIP sync due to getIntUnsynced)
        int[] viaToIntVector = argMaxResult.toIntVector();

        System.out.println("After GPU work: getInt(0)=" + viaGetInt +
                ", toIntVector[0]=" + viaToIntVector[0] +
                ", expected=" + expectedMaxIdx);

        assertEquals(expectedMaxIdx, viaGetInt,
                "getInt(0) should return " + expectedMaxIdx + " but got " + viaGetInt);
        assertEquals(expectedMaxIdx, viaToIntVector[0],
                "toIntVector()[0] should return " + expectedMaxIdx + " but got " + viaToIntVector[0]);
        assertEquals(viaGetInt, viaToIntVector[0],
                "getInt and toIntVector should agree: getInt=" + viaGetInt +
                        ", toIntVector=" + viaToIntVector[0]);
    }

    @Test
    public void testToIntVector_VsGetInt_AfterSameDiff() {
        // Same test but after SameDiff/DSP execution (closer to VLM context)
        int vocabSize = 49280;
        int hiddenSize = 576;

        org.nd4j.autodiff.samediff.SameDiff sd = org.nd4j.autodiff.samediff.SameDiff.create();
        org.nd4j.autodiff.samediff.SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, hiddenSize);
        org.nd4j.autodiff.samediff.SDVariable weight = sd.constant("weight", Nd4j.randn(DataType.FLOAT, hiddenSize, vocabSize));
        org.nd4j.autodiff.samediff.SDVariable output = sd.mmul("logits", input, weight);

        int failures = 0;
        for (int step = 0; step < 10; step++) {
            // Execute SameDiff graph (like decoder)
            INDArray inputArr = Nd4j.randn(DataType.FLOAT, 1, hiddenSize);
            java.util.Map<String, INDArray> results = sd.output(
                    java.util.Map.of("input", inputArr), "logits");
            INDArray logitsRaw = results.get("logits");  // [1, vocabSize]

            // Simulate VLM batch decode: extract last position, dup, argmax
            INDArray logits2d = logitsRaw.dup();  // [1, vocabSize]

            // argMax via dim path (batch path: argmaxBatch)
            INDArray argMaxResult = Nd4j.argMax(logits2d, 1);
            int[] viaToIntVector = argMaxResult.toIntVector();
            int viaGetInt = argMaxResult.getInt(0);

            // Also compare with sortWithIndices
            INDArray flat = logits2d.reshape(logits2d.length());
            INDArray[] sortResult = Nd4j.sortWithIndices(flat.dup(), 0, false);
            int sortArgmax = sortResult[0].getInt(0);

            // All three should agree
            boolean allAgree = (viaToIntVector[0] == viaGetInt) && (viaGetInt == sortArgmax);
            if (!allAgree) {
                failures++;
                System.out.println("Step " + step + " MISMATCH: toIntVector=" + viaToIntVector[0] +
                        ", getInt=" + viaGetInt + ", sort=" + sortArgmax +
                        ", val@getInt=" + flat.getFloat(viaGetInt) +
                        ", val@sort=" + flat.getFloat(sortArgmax));
            } else {
                System.out.println("Step " + step + " OK: all methods agree on index " + viaGetInt);
            }

            logits2d.close();
        }

        assertEquals(0, failures, failures + " out of 10 steps had mismatched argmax results");
    }

    @Test
    public void testToIntVector_RapidSuccession() {
        // Launch many argMax operations in rapid succession without explicit sync
        // This stresses the async pipeline and may reveal timing-dependent bugs
        int vocabSize = 49280;
        int numIterations = 50;
        int failures = 0;

        for (int i = 0; i < numIterations; i++) {
            // Random max position each iteration
            int expectedMaxIdx = (i * 997) % vocabSize;  // pseudo-random spread

            INDArray logits = Nd4j.zeros(DataType.FLOAT, 1, vocabSize);
            logits.putScalar(new int[]{0, expectedMaxIdx}, 100.0f);

            // Don't call commit() — let the async pipeline run
            INDArray argMaxResult = Nd4j.argMax(logits, 1);
            int[] result = argMaxResult.toIntVector();

            if (result[0] != expectedMaxIdx) {
                failures++;
                if (failures <= 5) {
                    System.out.println("Iter " + i + ": expected " + expectedMaxIdx +
                            ", got " + result[0] + " via toIntVector");
                }
            }

            logits.close();
            argMaxResult.close();
        }

        System.out.println("Rapid succession: " + (numIterations - failures) + "/" + numIterations + " correct");
        assertEquals(0, failures, failures + " out of " + numIterations +
                " rapid argMax+toIntVector calls returned wrong results");
    }

    @Test
    public void testArgMax_BlockBoundaries() {
        // Test at exact block boundaries (256 threads per block)
        int vocabSize = 49280;
        int blockSize = 256;

        for (int block = 0; block < vocabSize / blockSize; block++) {
            int maxIdx = block * blockSize;  // first element of each block
            INDArray logits = Nd4j.zeros(DataType.FLOAT, vocabSize);
            logits.putScalar(maxIdx, 10.0f);

            int result = Nd4j.argMax(logits).getInt(0);
            if (result != maxIdx) {
                System.out.println("Block " + block + " (idx=" + maxIdx + "): got " + result + " FAIL");
            }
            assertEquals(maxIdx, result,
                    "argMax failed at block boundary " + block + " (idx=" + maxIdx + ")");
        }
        System.out.println("All " + (vocabSize / blockSize) + " block boundaries passed");
    }
}
