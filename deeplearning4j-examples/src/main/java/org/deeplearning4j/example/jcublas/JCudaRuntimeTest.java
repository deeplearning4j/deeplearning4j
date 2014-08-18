
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
        int n = 500;
        int finagle = 78;
        int nn = n * (n + finagle);
        double[] b = new double[nn];
        for (int i = 0; i < nn; i++) {b[i] = 1;}
        double[] c = new double[nn];
        for (int i = 0; i < nn; i++) {c[i] = 2;}

        INDArray a0 = new JCublasNDArray(b,new int[]{n-finagle,n});
        INDArray a1 = new JCublasNDArray(c,new int[]{n,n-finagle});

        NDArray b0 = new NDArray(b,new int[]{n,n+finagle});
        NDArray b1 = new NDArray(c,new int[]{n+finagle,n});


        INDArray d0;
        NDArray d3;

        long start = System.nanoTime();
        d0 = a0.mmul(a1);
        System.out.printf("JCUBLAS (%dx%d * %dx%d --> %dx%d) started",a0.rows(), a0.columns(),
                a1.rows(), a1.columns(),
                d0.rows(), d0.columns());

        long time = System.nanoTime() - start;
        System.out.printf("JCUBLAS (%dx%d * %dx%d --> %dx%d) finished in %,d ns%n",
                a0.rows(), a0.columns(),
                a1.rows(), a1.columns(),
                d0.rows(), d0.columns(), time);

        start = System.nanoTime();
        d3 = b0.mmul(b1);
        time = System.nanoTime() - start;
        System.out.printf("Native CPU (%dx%d * %dx%d --> %dx%d) finished %,d ns%n",
                b0.rows(), b0.columns(),
                b1.rows(), b1.columns(),
                d3.rows(), d3.columns(), time);

        if (n < 20) {
            System.out.println("JCUblas output:");
            double[] d = d0.data();
            for (int i = 0; i < d.length; i++) {
                System.out.print(" " + Double.toString(d[i]));
            }

            System.out.println("\nNative CPU output:");
            double[] d_ = d3.data();
            for (int i = 0; i < d_.length; i++) {
                System.out.print(" " + Double.toString(d_[i]));
            }
            System.out.println("");
        }



        JCuda.cudaFree(pointer);
    }
}
