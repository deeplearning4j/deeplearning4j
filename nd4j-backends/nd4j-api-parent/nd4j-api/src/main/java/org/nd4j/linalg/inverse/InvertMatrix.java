package org.nd4j.linalg.inverse;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsoncccc on 11/30/15.
 */
public class InvertMatrix {


    /**
     * Inverts a matrix
     * @param arr the array to invert
     * @return the inverted matrix
     */
    public static INDArray invert(INDArray arr,boolean inPlace) {
        if (!arr.isSquare()) {
            throw new IllegalArgumentException("invalid array: must be square matrix");
        }

        int[] IPIV = new int[arr.length() + 1];
        int LWORK = arr.length() * arr.length();
        INDArray WORK = Nd4j.create(new double[LWORK]);
        INDArray inverse = inPlace ? arr : arr.dup();
        Nd4j.getBlasWrapper().lapack().getrf(arr.size(1),arr.size(0),inverse, arr.size(0),IPIV,0);
        Nd4j.getBlasWrapper().lapack().getri(arr.size(0),inverse,arr.size(0),IPIV,WORK,LWORK,0);
        return inverse;

    }

}
