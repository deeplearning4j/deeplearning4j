
package org.deeplearning4j.example.jcublas;

import jcuda.Pointer;
import jcuda.runtime.JCuda;

import org.deeplearning4j.linalg.*;

/**
 * Hello, World in JCuda
 * Taken from http://www.jcuda.org/tutorial/TutorialIndex.html
 */
public class JCudaRuntimeTest {
    public static void main(String args[]) {
        Pointer pointer = new Pointer();
        double[] b = new double[3];
	JCublasNDArray  a = new JCublasNDArray(b);
        JCuda.cudaMalloc(pointer, 4);
        System.out.println("Pointer: " + pointer);
        JCuda.cudaFree(pointer);
    }
}
