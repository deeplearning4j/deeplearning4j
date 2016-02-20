package org.nd4j.nativeblas;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.Buffer;

/**
 * Created by agibsonccc on 2/20/16.
 */
public interface PointerConverter {

    long toPointer(IComplexNDArray arr);

    long toPointer(INDArray arr);

    long toPointer(Buffer buffer);




}
