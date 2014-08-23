package org.deeplearning4j.linalg.api.io;

import org.deeplearning4j.linalg.api.complex.IComplexNDArray;
import org.deeplearning4j.linalg.api.ndarray.INDArray;

import java.io.*;

/**
 * Base class for NDArray IO
 *
 * @author Adam Gibson
 */
public abstract class BaseNDArrayIO implements NDArrayIO {

    /**
     * Write an ndarray to the output stream
     *
     * @param out the ndarray to write
     * @param to  the output stream to write to
     */
    @Override
    public void write(INDArray out, File to) throws IOException {
        FileOutputStream fos = new FileOutputStream(to);
        BufferedOutputStream bos = new BufferedOutputStream(fos);
        write(out,bos);
        bos.flush();
        bos.close();
    }

    /**
     * Write a complex ndarray to an output stream
     *
     * @param out the ndarray to write
     * @param to  the output stream to write to
     */
    @Override
    public void writeComplex(IComplexNDArray out, File to) throws IOException {
        FileOutputStream fos = new FileOutputStream(to);
        BufferedOutputStream bos = new BufferedOutputStream(fos);
        write(out,bos);
        bos.flush();
        bos.close();
    }

    /**
     * Read in an ndarray from a file
     *
     * @param file the file to read in from
     * @return the ndarray that was read in
     */
    @Override
    public INDArray read(File file) throws IOException {
        FileInputStream fis = new FileInputStream(file);
        INDArray ret = read(fis);
        fis.close();
        return ret;
    }

    /**
     * Read in a complex ndarray from a file
     *
     * @param file the ndarray to read from
     * @return the read in ndarray
     */
    @Override
    public IComplexNDArray readComplex(File file) throws IOException {
        FileInputStream fis = new FileInputStream(file);
        IComplexNDArray ret = readComplex(fis);
        fis.close();
        return ret;
    }
}
