
package org.deeplearning4j.example.jcublas;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import org.deeplearning4j.linalg.jcublas.JCublasNDArray;
import org.deeplearning4j.linalg.api.ndarray.INDArray;


/**
 * Hello, World in JCuda
 * Taken from http://www.jcuda.org/tutorial/TutorialIndex.html
 */
public class JCudaRuntimeTest {

    public static void main(String args[]) {



    }
    public void slices() {
        double[] data = JCublasNDArray.linspace(1,10,10).data;

        INDArray testMatrix = new JCublasNDArray(data,new int[]{5,2});
        INDArray row1 = testMatrix.getRow(0).transpose();
        INDArray row2 = testMatrix.getRow(1);
        JCublasNDArray row12 = JCublasNDArray.linspace(1,2,2).reshape(2,1);
        JCublasNDArray row22 = JCublasNDArray.linspace(3,4,2).reshape(1,2);
        JCublasNDArray rowResult = row12.mmul(row22);

        INDArray row122 = JCublasNDArray.wrap(row12);
        INDArray row222 = JCublasNDArray.wrap(row22);
        INDArray rowResult2 = row122.mmul(row222);

        INDArray mmul = row1.mmul(row2);
    }
    public void mmul() {
        Pointer pointer = new Pointer();
        int n = 500;
        int finagle = 0;
        int nn = n * (n + finagle);
        double[] b = new double[nn];
        for (int i = 0; i < nn; i++) {b[i] = 1;}
        double[] c = new double[nn];
        for (int i = 0; i < nn; i++) {c[i] = 2;}

        INDArray a0 = new JCublasNDArray(b,new int[]{n-finagle,n});
        INDArray a1 = new JCublasNDArray(c,new int[]{n,n-finagle});

        JCublasNDArray b0 = new JCublasNDArray(b,new int[]{n,n+finagle});
        JCublasNDArray b1 = new JCublasNDArray(c,new int[]{n+finagle,n});


        INDArray d0;
        JCublasNDArray d3;
        JCublasNDArray z;
        z = new JCublasNDArray(JCublasNDArray.ones(27).data,new int[]{3,3,3});

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
