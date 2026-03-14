package org.eclipse.deeplearning4j.dl4jcore.nn.multilayer;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.buffer.DataType;
import static org.junit.jupiter.api.Assertions.*;

public class TestMmulMinimal {
    @Test
    public void testMmulSmall() {
        // Reproduce backpropEpsilon: delta=[1,3] × w^T=[3,1] -> [1,1]
        INDArray delta = Nd4j.createFromArray(new float[][]{{-0.8953f, 0.2857f, 0.6097f}});
        INDArray w = Nd4j.createFromArray(new float[][]{{-1.3394f, -0.2077f, 0.6466f}});
        
        System.out.println("delta: " + delta + " shape=" + java.util.Arrays.toString(delta.shape()) + " stride=" + java.util.Arrays.toString(delta.stride()) + " order=" + delta.ordering());
        System.out.println("w: " + w + " shape=" + java.util.Arrays.toString(w.shape()) + " stride=" + java.util.Arrays.toString(w.stride()) + " order=" + w.ordering());
        
        INDArray wT = w.transpose();
        System.out.println("wT: " + wT + " shape=" + java.util.Arrays.toString(wT.shape()) + " stride=" + java.util.Arrays.toString(wT.stride()) + " order=" + wT.ordering());
        
        INDArray result = delta.mmul(wT);
        System.out.println("delta.mmul(wT) = " + result);
        
        // Manual: (-0.8953*-1.3394) + (0.2857*-0.2077) + (0.6097*0.6466)
        // = 1.1993 + (-0.0594) + 0.3942 = 1.5341
        float expected = -0.8953f*-1.3394f + 0.2857f*-0.2077f + 0.6097f*0.6466f;
        System.out.println("Expected: " + expected);
        assertEquals(expected, result.getFloat(0), 0.01f, "mmul result should match manual computation");
        
        // Also test with F-order w (as in the actual code)
        INDArray wF = w.dup('f');
        System.out.println("\nwF: " + wF + " shape=" + java.util.Arrays.toString(wF.shape()) + " stride=" + java.util.Arrays.toString(wF.stride()) + " order=" + wF.ordering());
        INDArray wFT = wF.transpose();
        System.out.println("wFT: " + wFT + " shape=" + java.util.Arrays.toString(wFT.shape()) + " stride=" + java.util.Arrays.toString(wFT.stride()) + " order=" + wFT.ordering());
        INDArray result2 = delta.mmul(wFT);
        System.out.println("delta.mmul(wFT) = " + result2);
        assertEquals(expected, result2.getFloat(0), 0.01f, "mmul with F-order w should also match");
        
        // Test with C-order delta with stride [1,1]
        // This replicates the exact scenario from the backprop
        INDArray deltaC = Nd4j.create(DataType.FLOAT, 1, 3).assign(delta);
        // Force stride [1,1] by creating as F then setting order?
        System.out.println("\ndeltaC: " + deltaC + " stride=" + java.util.Arrays.toString(deltaC.stride()) + " order=" + deltaC.ordering());
        INDArray result3 = deltaC.mmul(wFT);
        System.out.println("deltaC.mmul(wFT) = " + result3);
    }
}
