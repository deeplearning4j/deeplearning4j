package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.junit.jupiter.api.Assertions.*;

public class TestReduceMeanCuda {
    @Test
    public void testReduceMeanLastDimKeepdims() {
        // Test reduce_mean along last dim with keepdims=true
        // This is used in RMS norm: mean(x^2, dim=-1, keepdim=true)
        INDArray x = Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5, 6}).reshape(2, 3);
        INDArray mean = x.mean(true, -1);  // [2, 1]
        System.out.println("[ReduceMean] input=" + x);
        System.out.println("[ReduceMean] mean(dim=-1, keepdim=true)=" + mean);
        // Expected: row 0 mean = (1+2+3)/3 = 2.0, row 1 mean = (4+5+6)/3 = 5.0
        assertEquals(2.0f, mean.getFloat(0, 0), 1e-5f, "Row 0 mean");
        assertEquals(5.0f, mean.getFloat(1, 0), 1e-5f, "Row 1 mean");
    }

    @Test
    public void testRmsNormManual() {
        // Simulate RMS norm computation exactly as the model does it
        // x = [1, 2, 3, 4] (hidden_dim=4)
        // squared = x * x = [1, 4, 9, 16]
        // meanSquared = mean(squared, dim=-1, keepdim=true) = [7.5]
        // rms = sqrt(meanSquared + 1e-6) = sqrt(7.5000001) ≈ 2.7386
        // normalized = x / rms = [0.3651, 0.7303, 1.0954, 1.4606]
        INDArray x = Nd4j.createFromArray(new float[]{1, 2, 3, 4}).reshape(1, 1, 4);
        INDArray squared = x.mul(x);
        INDArray meanSquared = squared.mean(true, -1);
        INDArray rms = Nd4j.math.sqrt(meanSquared.add(1e-6));
        INDArray normalized = x.div(rms);

        System.out.println("[RmsNorm] x=" + x);
        System.out.println("[RmsNorm] squared=" + squared);
        System.out.println("[RmsNorm] meanSquared=" + meanSquared);
        System.out.println("[RmsNorm] rms=" + rms);
        System.out.println("[RmsNorm] normalized=" + normalized);

        float expectedMean = (1 + 4 + 9 + 16) / 4.0f;  // 7.5
        assertEquals(expectedMean, meanSquared.getFloat(0), 1e-4f, "meanSquared");

        float expectedRms = (float)Math.sqrt(7.5 + 1e-6);  // ~2.7386
        assertEquals(expectedRms, rms.getFloat(0), 1e-4f, "rms");

        // normalized[0] = 1 / 2.7386 ≈ 0.3651
        float expectedN0 = 1.0f / expectedRms;
        assertEquals(expectedN0, normalized.getFloat(0, 0, 0), 1e-4f, "normalized[0]");
    }

    @Test
    public void testRmsNormHiddenDim1024() {
        // Test at model-realistic dimension (hidden=1024)
        Nd4j.getRandom().setSeed(42);
        INDArray x = Nd4j.randn(DataType.FLOAT, 1, 7, 1024);  // [batch, seq, hidden]
        
        // RMS norm: x / sqrt(mean(x^2, dim=-1, keepdim=true) + eps)
        INDArray squared = x.mul(x);
        INDArray meanSquared = squared.mean(true, -1);  // [1, 7, 1]
        INDArray rms = Nd4j.math.sqrt(meanSquared.add(1e-6));
        INDArray normalized = x.div(rms);

        // The normalized values should have RMS close to 1.0
        // RMS of normalized = sqrt(mean(normalized^2)) ≈ 1.0
        INDArray normSquared = normalized.mul(normalized);
        INDArray normMeanSq = normSquared.mean(true, -1);  // should be ~1.0
        
        System.out.println("[RmsNorm1024] meanSquared stats: min=" + meanSquared.minNumber() + 
                          " max=" + meanSquared.maxNumber() + " mean=" + meanSquared.meanNumber());
        System.out.println("[RmsNorm1024] normMeanSq stats: min=" + normMeanSq.minNumber() + 
                          " max=" + normMeanSq.maxNumber() + " mean=" + normMeanSq.meanNumber());
        
        // After RMS normalization, the mean squared should be exactly 1.0 for each position
        for (int i = 0; i < 7; i++) {
            float val = normMeanSq.getFloat(0, i, 0);
            assertEquals(1.0f, val, 1e-4f, "RMS norm position " + i + " should give norm=1.0, got " + val);
        }
    }
}
