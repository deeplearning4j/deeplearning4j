package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by cvn on 9/7/14.
 */

public class VectorOperationExample {

    public static void main(String[] args) {

        INDArray nd = Nd4j.create(new float[]{1,2},new int[]{2}); //vector as row
        INDArray nd2 = Nd4j.create(new float[]{3,4},new int[]{2, 1}); //vector as column
        INDArray nd3 = Nd4j.create(new float[]{1,3,2,4},new int[]{2,2}); //elements arranged column major
        INDArray nd4 = Nd4j.create(new float[]{3,4,5,6},new int[]{2, 2});

        // Show initial matrices

        System.out.println(nd);
        System.out.println(nd2);
        System.out.println(nd3);

        //create nd-array variable to show result of nondestructive operations. matrix multiply row vector by column vector to obtain dot product.
        //assign product to nd-array variable.

        INDArray ndv = nd.mmul(nd2);

        System.out.println(ndv);

        //multiply a row by a 2 x 2 matrix

        ndv = nd.mmul(nd4);
        System.out.println(ndv);

        //multiply two 2 x 2 matrices

        ndv = nd3.mmul(nd4);
        System.out.println(ndv);

        //now switch the position of the matrices in the equation to obtain different result. matrix multiplication is not commutative.

        ndv = nd4.mmul(nd3);
        System.out.println(ndv);

        // switch the row and column vector to obtain the outer product

        ndv = nd2.mmul(nd);
        System.out.println(ndv);

        // let's see what happens if you double nd

        INDArray nd5 = Nd4j.create(new float[]{1,1,2,2},new int[]{2,2}); //doubling nd

        ndv = nd2.mmul(nd5);
        System.out.println(ndv); //same thing!


    }

}
