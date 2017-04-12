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

 DataBuffer getMinorPointer();
}
