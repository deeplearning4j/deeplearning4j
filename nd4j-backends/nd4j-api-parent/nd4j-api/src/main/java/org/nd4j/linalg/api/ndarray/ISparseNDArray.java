package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author Audrey Loeffel
 */
public interface ISparseNDArray extends INDArray {
    /*
    * TODO
    * Will contain methods such as toDense, toCSRFormat,...
    *
    * */

    /**
     * Return a array of non-major pointers
     * i.e. return the column indexes in case of row-major ndarray
     * @return a DataBuffer of indexes
     * */
    DataBuffer getVectorCoordinates();

    /**
     * Return a dense representation of the sparse ndarray
     * */
    INDArray toDense();

    /**
     * Return the number of non-null element
     * @return nnz
     * */
    int nnz();

    /**
     * Return the sparse format (i.e COO, CSR, ...)
     * @return format
     * @see SparseFormat
     * */
    SparseFormat getFormat();
}
