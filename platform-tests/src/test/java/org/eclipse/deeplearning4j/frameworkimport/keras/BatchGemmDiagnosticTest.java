package org.eclipse.deeplearning4j.frameworkimport.keras;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.nd4j.linalg.indexing.NDArrayIndex.*;

/**
 * Isolate batch gemm issues in LSTM processing.
 */
public class BatchGemmDiagnosticTest {

    @Test
    void testBatchGemmFOrder() {
        int M = 100;  // batch size
        int K = 128;  // nIn/nOut
        int N = 512;  // 4*nOut

        // Create A in F-order (simulates prevOutputActivations from LSTM)
        INDArray A_c = Nd4j.rand(M, K);
        INDArray A_f = A_c.dup('f');
        System.out.println("A_f shape=" + Arrays.toString(A_f.shape()) + " strides=" + Arrays.toString(A_f.stride()) + " order=" + A_f.ordering());
        System.out.println("A_c shape=" + Arrays.toString(A_c.shape()) + " strides=" + Arrays.toString(A_c.stride()) + " order=" + A_c.ordering());

        // Verify A_f and A_c have same data
        double aDiff = A_f.sub(A_c).maxNumber().doubleValue();
        System.out.println("A_f vs A_c maxDiff: " + aDiff);

        // Create B = recurrent weights [K, N]
        INDArray B = Nd4j.rand(K, N);

        // Test 1: mmul with F-order A
        INDArray result_f = A_f.mmul(B);
        INDArray result_c = A_c.mmul(B);
        double mmulDiff = result_f.sub(result_c).amaxNumber().doubleValue();
        System.out.println("\nTest 1 - mmul: A_f.mmul(B) vs A_c.mmul(B) maxDiff=" + mmulDiff);

        // Test 2: gemm with F-order A into existing C
        INDArray C1 = Nd4j.rand(M, N);
        INDArray C2 = C1.dup();
        Nd4j.gemm(A_f, B, C1, false, false, 1.0, 1.0);
        Nd4j.gemm(A_c, B, C2, false, false, 1.0, 1.0);
        double gemmDiff = C1.sub(C2).amaxNumber().doubleValue();
        System.out.println("Test 2 - gemm: F-order vs C-order A maxDiff=" + gemmDiff);

        // Test 3: Compare batch gemm with row-by-row gemm
        INDArray C3 = Nd4j.rand(M, N);
        INDArray C3_copy = C3.dup();
        Nd4j.gemm(A_f, B, C3, false, false, 1.0, 1.0);
        // Row-by-row
        for (int s = 0; s < M; s++) {
            INDArray a_row = A_f.get(point(s), all()).reshape(1, K);
            INDArray c_row = C3_copy.get(point(s), all()).reshape(1, N);
            Nd4j.gemm(a_row, B, c_row, false, false, 1.0, 1.0);
        }
        double rowVsBatchDiff = C3.sub(C3_copy).amaxNumber().doubleValue();
        System.out.println("Test 3 - batch vs row-by-row gemm: maxDiff=" + rowVsBatchDiff);

        // Test 4a: Check if mul has the same stride bug as sub
        System.out.println("\n=== Test 4a: mul with non-contiguous arrays ===");
        INDArray bigArray = Nd4j.rand(M, N);  // [100, 512]
        INDArray view1 = bigArray.get(all(), interval(0, K));  // [100, 128] strides [512, 1]
        INDArray contiguous1 = Nd4j.rand(M, K);  // [100, 128] strides [128, 1]
        System.out.println("view1 shape=" + Arrays.toString(view1.shape()) + " strides=" + Arrays.toString(view1.stride()));
        System.out.println("contiguous1 shape=" + Arrays.toString(contiguous1.shape()) + " strides=" + Arrays.toString(contiguous1.stride()));

        INDArray mulResult = contiguous1.mul(view1);  // like prevMemCellState.mul(forgetGateActivations)
        // Verify element-by-element
        int mulBugs = 0;
        for (int s = 0; s < M; s++) {
            for (int f = 0; f < K; f++) {
                float v1 = view1.getFloat(s, f);
                float c1 = contiguous1.getFloat(s, f);
                float expected2 = c1 * v1;
                float actual2 = mulResult.getFloat(s, f);
                if (Math.abs(expected2 - actual2) > 1e-5) {
                    mulBugs++;
                    if (mulBugs <= 3) {
                        System.out.printf("  MUL BUG [%d, %d]: c=%.6f v=%.6f expected=%.6f actual=%.6f%n",
                            s, f, c1, v1, expected2, actual2);
                    }
                }
            }
        }
        System.out.println("mul bugs: " + mulBugs + "/" + (M * K));

        // Also test in-place muli
        INDArray contiguous2 = contiguous1.dup();
        contiguous2.muli(view1);
        int muliBugs = 0;
        for (int s = 0; s < M; s++) {
            for (int f = 0; f < K; f++) {
                float v1 = view1.getFloat(s, f);
                float c1 = contiguous1.getFloat(s, f);
                float expected2 = c1 * v1;
                float actual2 = contiguous2.getFloat(s, f);
                if (Math.abs(expected2 - actual2) > 1e-5) {
                    muliBugs++;
                }
            }
        }
        System.out.println("muli bugs: " + muliBugs + "/" + (M * K));

        // Test addi between view and contiguous
        INDArray contiguous3 = Nd4j.rand(M, K);
        INDArray contiguous3copy = contiguous3.dup();
        contiguous3.addi(view1);
        int addiBugs = 0;
        for (int s = 0; s < M; s++) {
            for (int f = 0; f < K; f++) {
                float v1 = view1.getFloat(s, f);
                float c1 = contiguous3copy.getFloat(s, f);
                float expected2 = c1 + v1;
                float actual2 = contiguous3.getFloat(s, f);
                if (Math.abs(expected2 - actual2) > 1e-5) {
                    addiBugs++;
                }
            }
        }
        System.out.println("addi bugs: " + addiBugs + "/" + (M * K));

        // Test in-place on view: view.muli(contiguous)
        INDArray bigArray2 = bigArray.dup();
        INDArray view2 = bigArray2.get(all(), interval(0, K));  // [100, 128] strides [512, 1]
        view2.muli(contiguous1);
        int viewMuliBugs = 0;
        for (int s = 0; s < M; s++) {
            for (int f = 0; f < K; f++) {
                float v1 = bigArray.getFloat(s, f);  // original view value
                float c1 = contiguous1.getFloat(s, f);
                float expected2 = v1 * c1;
                float actual2 = view2.getFloat(s, f);
                if (Math.abs(expected2 - actual2) > 1e-5) {
                    viewMuliBugs++;
                    if (viewMuliBugs <= 3) {
                        System.out.printf("  VIEW MULI BUG [%d, %d]: orig=%.6f c=%.6f expected=%.6f actual=%.6f%n",
                            s, f, v1, c1, expected2, actual2);
                    }
                }
            }
        }
        System.out.println("view.muli(contiguous) bugs: " + viewMuliBugs + "/" + (M * K));

        // Test 4: Simulate LSTM per-timestep input extraction from permuted NWC
        INDArray nwcInput = Nd4j.rand(M, 80, K);  // [batch, timesteps, features] NWC
        INDArray ncwInput = nwcInput.permute(0, 2, 1);  // [batch, features, timesteps] NCW
        System.out.println("\nNCW input shape=" + Arrays.toString(ncwInput.shape()) + " strides=" + Arrays.toString(ncwInput.stride()));

        INDArray timestep0 = ncwInput.tensorAlongDimension(0, 1, 0);
        System.out.println("timestep0 shape=" + Arrays.toString(timestep0.shape()) + " strides=" + Arrays.toString(timestep0.stride()) + " offset=" + timestep0.offset());

        // Copy via toMmulCompatible
        INDArray timestep0Copy = org.nd4j.linalg.api.shape.Shape.toMmulCompatible(timestep0);
        System.out.println("timestep0Copy shape=" + Arrays.toString(timestep0Copy.shape()) + " strides=" + Arrays.toString(timestep0Copy.stride()));

        // Verify data integrity of copy - test both sub and element-by-element
        double copyDiffDirect = timestep0.sub(timestep0Copy).amaxNumber().doubleValue();
        double copyDiffDuped = timestep0.dup().sub(timestep0Copy).amaxNumber().doubleValue();
        System.out.println("timestep0 copy diff (direct sub, may be buggy): " + copyDiffDirect);
        System.out.println("timestep0 copy diff (dup'd sub): " + copyDiffDuped);

        // Element-by-element comparison
        int elemMismatches = 0;
        for (int s = 0; s < M; s++) {
            for (int f = 0; f < K; f++) {
                float expected = timestep0.getFloat(s, f);
                float actual = timestep0Copy.getFloat(s, f);
                if (Math.abs(expected - actual) > 1e-6) {
                    elemMismatches++;
                    if (elemMismatches <= 3) {
                        System.out.printf("  ELEM MISMATCH [%d, %d]: src=%.6f copy=%.6f%n", s, f, expected, actual);
                    }
                }
            }
        }
        System.out.println("Element mismatches in copy: " + elemMismatches + "/" + (M * K));

        // Check if sub between non-contiguous arrays is buggy
        INDArray subResult = timestep0.sub(timestep0Copy);
        System.out.println("Sub result shape=" + Arrays.toString(subResult.shape()) + " strides=" + Arrays.toString(subResult.stride()));
        // Check specific elements of sub result
        int subBugs = 0;
        for (int s = 0; s < M; s++) {
            for (int f = 0; f < K; f++) {
                float srcVal = timestep0.getFloat(s, f);
                float copyVal = timestep0Copy.getFloat(s, f);
                float subVal = subResult.getFloat(s, f);
                float expectedDiff = srcVal - copyVal;
                if (Math.abs(subVal - expectedDiff) > 1e-6) {
                    subBugs++;
                    if (subBugs <= 3) {
                        System.out.printf("  SUB BUG [%d, %d]: src=%.6f copy=%.6f expectedDiff=%.6f actualSubResult=%.6f%n",
                            s, f, srcVal, copyVal, expectedDiff, subVal);
                    }
                }
            }
        }
        System.out.println("Sub operation bugs: " + subBugs + "/" + (M * K));

        // Test 5: Full LSTM-like batch computation
        System.out.println("\n=== Test 5: LSTM-like batch computation ===");
        INDArray inputWeights = Nd4j.rand(K, N);
        INDArray recurrentWeights = Nd4j.rand(K, N);
        INDArray biases = Nd4j.rand(1, N);

        // First timestep
        INDArray ts0 = org.nd4j.linalg.api.shape.Shape.toMmulCompatible(ncwInput.tensorAlongDimension(0, 1, 0));
        INDArray ifog_batch = ts0.mmul(inputWeights);
        ifog_batch.addiRowVector(biases);

        // Compare with single-sample computation
        int maxDiffSamples = 0;
        for (int s = 0; s < M; s++) {
            INDArray x_s = nwcInput.get(point(s), point(0), all()).reshape(1, K);
            INDArray ifog_single = x_s.mmul(inputWeights).addiRowVector(biases);
            INDArray batch_s = ifog_batch.get(point(s), all()).reshape(1, N);
            double diff = ifog_single.sub(batch_s).amaxNumber().doubleValue();
            if (diff > 1e-5) {
                maxDiffSamples++;
                if (maxDiffSamples <= 3) {
                    System.out.printf("  Timestep 0 mismatch sample %d: maxDiff=%.8f%n", s, diff);
                }
            }
        }
        System.out.println("Timestep 0 batch vs single: " + maxDiffSamples + " mismatched samples");

        // Second timestep with F-order prevOutput
        INDArray prevH = Nd4j.rand(M, K).dup('f');  // F-order like LSTM produces
        INDArray ts1 = org.nd4j.linalg.api.shape.Shape.toMmulCompatible(ncwInput.tensorAlongDimension(1, 1, 0));
        INDArray ifog_batch2 = ts1.mmul(inputWeights);
        Nd4j.gemm(prevH, recurrentWeights, ifog_batch2, false, false, 1.0, 1.0);
        ifog_batch2.addiRowVector(biases);

        maxDiffSamples = 0;
        for (int s = 0; s < M; s++) {
            INDArray x_s = nwcInput.get(point(s), point(1), all()).reshape(1, K);
            INDArray h_s = prevH.get(point(s), all()).reshape(1, K);
            INDArray ifog_single = x_s.mmul(inputWeights);
            Nd4j.gemm(h_s, recurrentWeights, ifog_single, false, false, 1.0, 1.0);
            ifog_single.addiRowVector(biases);
            INDArray batch_s = ifog_batch2.get(point(s), all()).reshape(1, N);
            double diff = ifog_single.sub(batch_s).amaxNumber().doubleValue();
            if (diff > 1e-5) {
                maxDiffSamples++;
                if (maxDiffSamples <= 3) {
                    System.out.printf("  Timestep 1 mismatch sample %d: maxDiff=%.8f%n", s, diff);
                    System.out.printf("    h_s shape=%s strides=%s order=%c%n",
                        Arrays.toString(h_s.shape()), Arrays.toString(h_s.stride()), h_s.ordering());
                }
            }
        }
        System.out.println("Timestep 1 batch (F-order prevH) vs single: " + maxDiffSamples + " mismatched samples");
    }
}
