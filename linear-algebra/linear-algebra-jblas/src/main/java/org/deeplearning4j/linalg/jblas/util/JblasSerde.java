package org.deeplearning4j.linalg.jblas.util;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.jblas.NDArray;
import org.jblas.DoubleMatrix;

import java.io.DataInputStream;
import java.io.IOException;

/**
 * Handles loading and saving of jblas matrices
 *
 * @author Adam Gibson
 */
public class JblasSerde {

    public static INDArray fromAscii(String path) throws IOException {
        return new NDArray(DoubleMatrix.loadAsciiFile(path));
    }

    /**
     * Read in the jblas binary format
     * @param dataInputStream the data inputstream to use
     * @return an ndarray of the same shape
     * @throws IOException
     */
    public static INDArray readJblasBinary(DataInputStream dataInputStream) throws IOException{
        DoubleMatrix d = new DoubleMatrix();
        d.in(dataInputStream);
        return new NDArray(d);

    }


}
