package org.nd4j.linalg.api.blas;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Blas buffer util for interopping with the underlying buffers
 * and the given ndarrays
 *
 * @author Adam Gibson
 */
public class BlasBufferUtil {

    /**
     * Get blas stride for the
     * given array
     * @param arr the array
     * @return the blas stride
     */
    public static int getBlasStride(INDArray arr) {
        if(arr instanceof IComplexNDArray)
            return arr.majorStride() / 2;
        return arr.majorStride();
    }

    /**
     * Get blas stride for the
     * given array
     * @param arr the array
     * @return the blas stride
     */
    public static int getBlasOffset(INDArray arr) {
        if(arr instanceof IComplexNDArray)
            return arr.offset() / 2;
        return arr.offset();

    }

    public static float[] getFloatData(INDArray buf) {
        if(buf.data().allocationMode() == DataBuffer.AllocationMode.HEAP) {
            if(buf.length() == buf.data().length() && buf.offset() == 0)
                return buf.data().asFloat();
            else {
                float[] ret = new float[buf.length()];
                for(int i = 0; i < buf.length(); i++)
                    ret[i] = buf.getFloat(i);
                return ret;
            }
        }
        else {
            float[] ret = new float[buf.length()];
            for(int i = 0; i < buf.length(); i++)
                ret[i] = buf.getFloat(i);
            return ret;
        }
    }

    public static double[] getDoubleData(INDArray buf) {
        if(buf.data().allocationMode() == DataBuffer.AllocationMode.HEAP) {
            if(buf.length() == buf.data().length() && buf.offset() == 0)
                return buf.data().asDouble();
            else {
                double[] ret = new double[buf.length()];
                for(int i = 0; i < buf.length(); i++)
                    ret[i] = buf.getDouble(i);
                return ret;
            }
        }
        else {
            double[] ret = new double[buf.length()];
            for(int i = 0; i < buf.length(); i++)
                ret[i] = buf.getDouble(i);
            return ret;

        }
    }

    public static float[] getFloatData(DataBuffer buf) {
        if(buf.allocationMode() == DataBuffer.AllocationMode.HEAP) {
            return buf.asFloat();
        }
        else {
            float[] ret = new float[buf.length()];
            for(int i = 0; i < buf.length(); i++)
                ret[i] = buf.getFloat(i);
            return ret;
        }
    }

    public static double[] getDoubleData(DataBuffer buf) {
        if(buf.allocationMode() == DataBuffer.AllocationMode.HEAP)
            return buf.asDouble();
        else {
            double[] ret = new double[buf.length()];
            for(int i = 0; i < buf.length(); i++)
                ret[i] = buf.getDouble(i);
            return ret;

        }
    }
}
