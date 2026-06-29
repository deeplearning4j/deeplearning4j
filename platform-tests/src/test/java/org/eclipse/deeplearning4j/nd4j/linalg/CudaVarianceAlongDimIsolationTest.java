package org.eclipse.deeplearning4j.nd4j.linalg;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.summarystats.StandardDeviation;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Isolation test: does CUDA varianceAlongDimension (SummaryStatsVariance) return ≈0 for TAD 0?
 *
 * Input: [[1,2,3],[4,5,6]] shape [2,3]
 * Expected population variance along axis=1: row0 = 2/3 ≈ 0.6667, row1 = 2/3 ≈ 0.6667
 * Expected population std dev along axis=1: row0 = sqrt(2/3) ≈ 0.8165, row1 ≈ 0.8165
 *
 * The bug reported: row 0 (TAD 0) returns ≈0 while row 1 returns ≈0.6667.
 *
 * DO NOT FIX — ISOLATION ONLY.
 */
public class CudaVarianceAlongDimIsolationTest {

    private static final double EXPECTED_POPULATION_VARIANCE = 2.0 / 3.0;  // ≈ 0.6667
    private static final double EXPECTED_POPULATION_STDDEV   = Math.sqrt(2.0 / 3.0); // ≈ 0.8165
    private static final double TOLERANCE = 1e-5;

    @Test
    public void testVarianceAlongDim1PopulationRepeated() {
        System.out.println("=== CudaVarianceAlongDimIsolationTest: variance along dim 1, 5 runs ===");
        System.out.println("Backend: " + Nd4j.getBackend().getClass().getName());
        System.out.println("Expected population variance per row: " + EXPECTED_POPULATION_VARIANCE);
        System.out.println();

        // Run 5 times to check for race / flakiness
        for (int run = 1; run <= 5; run++) {
            System.out.println("--- Run " + run + " ---");

            // [[1,2,3],[4,5,6]] as DOUBLE to match layer_norm_bp usage
            INDArray x = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, new long[]{2, 3}, DataType.DOUBLE);
            System.out.println("Input x (shape=" + java.util.Arrays.toString(x.shape()) + "):");
            System.out.println(x.toStringFull());

            // --- Variance (biasCorrected=false = population variance) ---
            Variance varOp = new Variance(x, false, 1L);  // biasCorrected=false, dim=1
            INDArray varResult = Nd4j.getExecutioner().exec(varOp);
            System.out.println("Variance result (shape=" + java.util.Arrays.toString(varResult.shape()) + "):");
            System.out.println(varResult.toStringFull());

            double var0 = varResult.getDouble(0);
            double var1 = varResult.getDouble(1);
            System.out.printf("  var[0] = %.8f  (expected %.8f, diff=%.2e)%n", var0, EXPECTED_POPULATION_VARIANCE, Math.abs(var0 - EXPECTED_POPULATION_VARIANCE));
            System.out.printf("  var[1] = %.8f  (expected %.8f, diff=%.2e)%n", var1, EXPECTED_POPULATION_VARIANCE, Math.abs(var1 - EXPECTED_POPULATION_VARIANCE));

            boolean var0ok = Math.abs(var0 - EXPECTED_POPULATION_VARIANCE) < TOLERANCE;
            boolean var1ok = Math.abs(var1 - EXPECTED_POPULATION_VARIANCE) < TOLERANCE;
            System.out.println("  var[0] correct: " + var0ok + "  var[1] correct: " + var1ok);
            if (!var0ok) {
                System.out.println("  *** BUG REPRODUCED: var[0] (TAD 0) incorrect ***");
            }

            // --- Standard deviation (biasCorrected=false = population std dev) ---
            INDArray x2 = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, new long[]{2, 3}, DataType.DOUBLE);
            StandardDeviation stdOp = new StandardDeviation(x2, false, 1L);  // biasCorrected=false, dim=1
            INDArray stdResult = Nd4j.getExecutioner().exec(stdOp);
            System.out.println("StdDev result (shape=" + java.util.Arrays.toString(stdResult.shape()) + "):");
            System.out.println(stdResult.toStringFull());

            double std0 = stdResult.getDouble(0);
            double std1 = stdResult.getDouble(1);
            System.out.printf("  std[0] = %.8f  (expected %.8f, diff=%.2e)%n", std0, EXPECTED_POPULATION_STDDEV, Math.abs(std0 - EXPECTED_POPULATION_STDDEV));
            System.out.printf("  std[1] = %.8f  (expected %.8f, diff=%.2e)%n", std1, EXPECTED_POPULATION_STDDEV, Math.abs(std1 - EXPECTED_POPULATION_STDDEV));

            boolean std0ok = Math.abs(std0 - EXPECTED_POPULATION_STDDEV) < TOLERANCE;
            boolean std1ok = Math.abs(std1 - EXPECTED_POPULATION_STDDEV) < TOLERANCE;
            System.out.println("  std[0] correct: " + std0ok + "  std[1] correct: " + std1ok);
            System.out.println();
        }

        // Final assertions: assert BOTH rows are correct on var AND std
        // (If these fail the test reports the bug; check the printout above for which run/row failed)
        INDArray xFinal = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, new long[]{2, 3}, DataType.DOUBLE);
        Variance varFinal = new Variance(xFinal, false, 1L);
        INDArray varFinalResult = Nd4j.getExecutioner().exec(varFinal);

        System.out.println("=== FINAL ASSERTION (run 6) ===");
        System.out.println("Variance:");
        System.out.println(varFinalResult.toStringFull());

        assertEquals(EXPECTED_POPULATION_VARIANCE, varFinalResult.getDouble(0), TOLERANCE,
                "CUDA variance TAD 0 should be ≈0.6667, got " + varFinalResult.getDouble(0));
        assertEquals(EXPECTED_POPULATION_VARIANCE, varFinalResult.getDouble(1), TOLERANCE,
                "CUDA variance TAD 1 should be ≈0.6667, got " + varFinalResult.getDouble(1));

        INDArray xFinal2 = Nd4j.create(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, new long[]{2, 3}, DataType.DOUBLE);
        StandardDeviation stdFinal = new StandardDeviation(xFinal2, false, 1L);
        INDArray stdFinalResult = Nd4j.getExecutioner().exec(stdFinal);

        System.out.println("StdDev:");
        System.out.println(stdFinalResult.toStringFull());

        assertEquals(EXPECTED_POPULATION_STDDEV, stdFinalResult.getDouble(0), TOLERANCE,
                "CUDA stddev TAD 0 should be ≈0.8165, got " + stdFinalResult.getDouble(0));
        assertEquals(EXPECTED_POPULATION_STDDEV, stdFinalResult.getDouble(1), TOLERANCE,
                "CUDA stddev TAD 1 should be ≈0.8165, got " + stdFinalResult.getDouble(1));

        System.out.println("=== DONE ===");
    }
}
