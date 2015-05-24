package org.nd4j.linalg.util;

import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrayFactory;

/**
 *
 * Util class for handling edge cases where linear view
 * is desirable but not usable (gpus)
 *
 * @author Adam Gibson
 */
public class LinearUtil {

    /**
     * When needing to look at an array that's a matrix or some other tensor
     * we typically need to map those arrays element wise (take axpy or uploading data as a linear buffer for example)
     * This handles finding the proper offset for the situations where a >= 2d array needs to be interpreted in a linear
     * form that may not be used outside of those contexts.
     *
     * Note that this is different than linear view in that while this does technically interpret the buffer as a linear
     * element wise buffer, this does not preserve structure which maybe needed for some operations.
     *
     * This is purely for operations like uploading to the gpu or element wise operations like axpy
     * @param arr  the array to get the stride for
     * @return the linear stride (USED ONLY FOR AXPY AND ELEMENT WISE OPERATIONS WHERE THE BUFFER IS DIRECTLY INVOLVED)
     * for this ndarray
     */
    public static int linearStride(INDArray arr) {
        if(arr.ordering() == NDArrayFactory.C && arr.isMatrix() && arr.stride(0) == 1)
            return arr.stride(1);
        return
                arr.offset() == 0
                        && !arr.isRowVector()
                        && !arr.isScalar() ?  arr.elementStride() : arr.majorStride();
    }

}
