package org.eclipse.deeplearning4j.nd4j.linalg;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class MeanDebug {
    @Test
    public void debugMeanComputation() {
        INDArray arr = Nd4j.linspace(1,12,12, DataType.DOUBLE).reshape('c',3,4);
        INDArray subView = arr.get(NDArrayIndex.interval(1,3), NDArrayIndex.interval(1,4));
        
        System.out.println("subView shape: " + java.util.Arrays.toString(subView.shape()));
        System.out.println("subView length: " + subView.length());
        
        // Compare different ways to compute mean
        double sum = subView.sumNumber().doubleValue();
        System.out.println("\nSum (sumNumber): " + sum);
        
        double meanFromSumAndLength = sum / subView.length();
        System.out.println("Manual mean (sum/length): " + meanFromSumAndLength);
        
        // Try mean with explicit dimensions
        System.out.println("\nmean(): " + subView.mean());
        System.out.println("mean(0,1): " + subView.mean(0,1));
        
        // Check dup
        INDArray dupView = subView.dup();
        System.out.println("\ndupView mean(): " + dupView.mean());
    }
}
