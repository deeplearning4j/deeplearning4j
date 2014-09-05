package org.nd4j.linalg.jblas.io;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.io.BaseNDArrayIO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrays;
import org.nd4j.linalg.jblas.util.JblasSerde;

import java.io.*;

/**
 * Jblas NDArray IO
 *
 * @author Adam Gibson
 */
public class JblasNDArrayIO extends BaseNDArrayIO {
    /**
     * Read in an ndrray from an input stream
     *
     * @param is the input stream to read in from
     * @return the ndarray read in
     */
    @Override
    public INDArray read(InputStream is) throws IOException {
        return JblasSerde.readJblasBinary(new DataInputStream(is));
    }

    /**
     * Read in a complex ndarray
     *
     * @param is the input stream to read in from
     * @return the complex ndarray that was read in
     */
    @Override
    public IComplexNDArray readComplex(InputStream is) throws IOException {
        return NDArrays.createComplex(read(is));
    }

    /**
     * Write an ndarray to the output stream
     *
     * @param out the ndarray to write
     * @param to  the output stream to write to
     */
    @Override
    public void write(INDArray out, OutputStream to) throws IOException {
          NDArrays.write(out,new DataOutputStream(to));
    }

    /**
     * Write a complex ndarray to an output stream
     *
     * @param out the ndarray to write
     * @param to  the output stream to write to
     */
    @Override
    public void writeComplex(IComplexNDArray out, OutputStream to) throws IOException {
            NDArrays.writeComplex(out,new DataOutputStream(to));
    }
}
