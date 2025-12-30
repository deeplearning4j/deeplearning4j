package org.eclipse.deeplearning4j.nd4j.linalg;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DebugGetColumns {
    public static void main(String[] args) {
        INDArray arr = Nd4j.linspace(1, 24, 24, DataType.DOUBLE).reshape(4, 6);
        System.out.println("Original array shape: " + java.util.Arrays.toString(arr.shape()));
        System.out.println("Original array ordering: " + arr.ordering());
        System.out.println("Original array:\n" + arr);
        
        System.out.println("\ngetColumn(0):");
        INDArray col0 = arr.getColumn(0);
        System.out.println("Shape: " + java.util.Arrays.toString(col0.shape()));
        System.out.println("Values: " + col0);
        
        System.out.println("\ngetColumn(1):");
        INDArray col1 = arr.getColumn(1);
        System.out.println("Shape: " + java.util.Arrays.toString(col1.shape()));
        System.out.println("Values: " + col1);
        
        System.out.println("\ngetColumns(0, 1):");
        INDArray cols = arr.getColumns(0, 1);
        System.out.println("Shape: " + java.util.Arrays.toString(cols.shape()));
        System.out.println("Values:\n" + cols);
        
        System.out.println("\nExpected: [[1, 5], [2, 6], [3, 7], [4, 8]]");
    }
}
