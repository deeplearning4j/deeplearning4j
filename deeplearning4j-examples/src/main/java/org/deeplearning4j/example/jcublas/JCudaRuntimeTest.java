
package org.deeplearning4j.example.jcublas;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import org.deeplearning4j.linalg.JCublasNDArray;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.linalg.NDArray;

/**
 * Hello, World in JCuda
 * Taken from http://www.jcuda.org/tutorial/TutorialIndex.html
 */
public class JCudaRuntimeTest {
    public static void main(String args[]) {
        Pointer pointer = new Pointer();
        int n = 10;
        int[] shape = new int[2];
        shape[0] = n;
        shape[1] = n;
        double[] b = new double[n*n];
        for (int i = 0; i < n*n; i++) {b[i] = 1;}
        double[] c = new double[n*n];
        for (int i = 0; i < n*n; i++) {c[i] = 2;}

        INDArray a0 = new JCublasNDArray(n,n,b);
        INDArray a1 = new JCublasNDArray(n,n,c);

        NDArray b0 = new NDArray(b,new int[]{n,n});
        NDArray b1 = new NDArray(c,new int[]{n,n});

        INDArray d1 = new JCublasNDArray(new double[]{1.0},new int[]{1,1});
        INDArray d2 = new JCublasNDArray(new double[]{1.0},new int[]{1,1});
        INDArray e1 = new JCublasNDArray(2,2,new double[]{1,2,3,4});
        INDArray e2 = new JCublasNDArray(2,2,new double[]{1,2,3,4});
        INDArray d0 = d1.mmul(d2);
        NDArray d3;

        d0 = a0.mmul(a1);
        System.out.println("JCUBLAS finished");
        d3 = b0.mmul(b1);
        System.out.println("Native CPU finished");

        System.out.println("JCUblas output:");
        double[] d = d0.data();
        for (int i = 0; i < d.length; i++) {System.out.print (" " + Double.toString(d[i]));}
        System.out.println("\nNative CPU output:");
        double[] d_ = d3.data();
        for (int i = 0; i < d_.length; i++) {System.out.print(" " + Double.toString(d_[i]));}
        System.out.println("");



        JCuda.cudaFree(pointer);
    }
}
