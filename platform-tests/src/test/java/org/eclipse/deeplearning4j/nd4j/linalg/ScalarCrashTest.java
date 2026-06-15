package org.eclipse.deeplearning4j.nd4j.linalg;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ScalarCrashTest {

    @Test
    public void testScalarCreation() {
        System.out.println("Creating scalar...");
        INDArray scalar = Nd4j.scalar(5.0);
        System.out.println("Scalar created: " + scalar);
        System.out.println("Scalar shape: " + java.util.Arrays.toString(scalar.shape()));
        System.out.println("Scalar length: " + scalar.length());
    }

    @Test
    public void testScalarSum() {
        System.out.println("Creating scalar...");
        INDArray scalar = Nd4j.scalar(5.0);
        System.out.println("Scalar created");

        System.out.println("Calling sum()...");
        INDArray result = scalar.sum();
        System.out.println("Sum result: " + result);
        System.out.println("Sum value: " + result.getDouble(0));
    }

    @Test
    public void testScalarMean() {
        System.out.println("Creating scalar...");
        INDArray scalar = Nd4j.scalar(5.0);
        System.out.println("Scalar created");

        System.out.println("Calling mean()...");
        INDArray result = scalar.mean();
        System.out.println("Mean result: " + result);
        System.out.println("Mean value: " + result.getDouble(0));
    }
}
