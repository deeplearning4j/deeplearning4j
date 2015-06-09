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
    public static int getBlasOffset(INDArray arr) {
        return arr.offset();
    }

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
     * Returns the float data
     * for this ndarray.
     * If possible (the offset is 0 representing the whole buffer)
     * it will return a direct reference to the underlying array
     * @param buf the ndarray to get the data for
     * @return the float data for this ndarray
     */
    public static float[] getFloatData(INDArray buf) {
        if(buf.data().dataType() != DataBuffer.Type.FLOAT)
            throw new IllegalArgumentException("Float data must be obtained from a float buffer");

        if(buf.data().allocationMode() == DataBuffer.AllocationMode.HEAP) {
            if(buf.length() == buf.data().length() && buf.offset() == 0)
                return buf.data().asFloat();
            else {
                INDArray linear = buf.linearView();
                float[] ret = new float[buf.length()];
                for(int i = 0; i < buf.length(); i++)
                    ret[i] = linear.getFloat(i);
                return ret;
            }
        }
        else {
            float[] ret = new float[buf.length()];
            INDArray linear = buf.linearView();

            for(int i = 0; i < buf.length(); i++)
                ret[i] = linear.getFloat(i);
            return ret;
        }
    }

    /**
     * Returns the double data
     * for this ndarray.
     * If possible (the offset is 0 representing the whole buffer)
     * it will return a direct reference to the underlying array
     * @param buf the ndarray to get the data for
     * @return the double data for this ndarray
     */
    public static double[] getDoubleData(INDArray buf) {
        if(buf.data().dataType() != DataBuffer.Type.DOUBLE)
            throw new IllegalArgumentException("Double data must be obtained from a double buffer");

        if(buf.data().allocationMode() == DataBuffer.AllocationMode.HEAP) {
            if(buf.length() == buf.data().length() && buf.offset() == 0)
                return buf.data().asDouble();
            else {
                double[] ret = new double[buf.length()];
                INDArray linear = buf.linearView();
                for(int i = 0; i < buf.length(); i++)
                    ret[i] = linear.getDouble(i);
                return ret;
            }
        }
        else {
            double[] ret = new double[buf.length()];
            INDArray linear = buf.linearView();
            for(int i = 0; i < buf.length(); i++)
                ret[i] = linear.getDouble(i);
            return ret;

        }
    }

    /**
     * Returns the float data
     * for this buffer.
     * If possible (the offset is 0 representing the whole buffer)
     * it will return a direct reference to the underlying array
     * @param buf the ndarray to get the data for
     * @return the double data for this ndarray
     */
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

    /**
     * Returns the double data
     * for this buffer.
     * If possible (the offset is 0 representing the whole buffer)
     * it will return a direct reference to the underlying array
     * @param buf the ndarray to get the data for
     * @return the double data for this buffer
     */
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


    /**
     * Set the data for the underlying array.
     * If the underlying buffer's array is equivalent to this array
     * it returns (avoiding an unneccessary copy)
     *
     * If the underlying storage mechanism isn't heap (no arrays)
     * it just copied the data over (strided access with offsets where neccessary)
     *
     * This is meant to be used with blas operations where the underlying blas implementation
     * takes an array but the data buffer being used might not be an array.
     *
     * This is also for situations where there is strided access and it's not
     * optimal to want to use the whole data buffer but just the subset of the
     * buffer needed for calculations.
     *
     *
     * @param data the data to set
     * @param toSet the array to set the data to
     */
    public static void setData(float[] data,INDArray toSet) {
        if(toSet.data().dataType() != DataBuffer.Type.FLOAT) {
            throw new IllegalArgumentException("Unable to set double data for type " + toSet.data().dataType());
        }

        if(toSet.data().allocationMode() == DataBuffer.AllocationMode.HEAP) {
            Object array = toSet.data().array();
            //data is assumed to have already been updated
            if(array == data)
                return;
            else {
                //copy the data over directly to the underlying array
                float[] d = (float[]) array;

                if(toSet.offset() == 0 && toSet.length() == data.length)
                    System.arraycopy(data,0,d,0,d.length);
                else {
                    int count = 0;
                    //need to do strided access with offset
                    for(int i = 0; i < data.length; i++) {
                        int dIndex = toSet.offset() + (i * toSet.majorStride());
                        d[dIndex] = data[count++];
                    }
                }
            }
        }
        else {
            //assumes the underlying data is in the right order
            DataBuffer underlyingData = toSet.data();
            if(data.length == toSet.length() && toSet.offset() == 0) {
                for(int i = 0; i < toSet.length(); i++) {
                    underlyingData.put(i,data[i]);
                }
            }
            else {
                int count = 0;
                //need to do strided access with offset
                for(int i = 0; i < data.length; i++) {
                    int dIndex = toSet.offset() + (i * toSet.majorStride());
                    underlyingData.put(dIndex,data[count++]);
                }
            }
        }

    }

    /**
     * Set the data for the underlying array.
     * If the underlying buffer's array is equivalent to this array
     * it returns (avoiding an unneccessary copy)
     *
     * If the underlying storage mechanism isn't heap (no arrays)
     * it just copied the data over (strided access with offsets where neccessary)
     *
     * This is meant to be used with blas operations where the underlying blas implementation
     * takes an array but the data buffer being used might not be an array.
     *
     * This is also for situations where there is strided access and it's not
     * optimal to want to use the whole data buffer but just the subset of the
     * buffer needed for calculations.
     *
     *
     * @param data the data to set
     * @param toSet the array to set the data to
     */
    public static void setData(double[] data,INDArray toSet) {
        if(toSet.data().dataType() != DataBuffer.Type.DOUBLE) {
            throw new IllegalArgumentException("Unable to set double data for type " + toSet.data().dataType());
        }

        if(toSet.data().allocationMode() == DataBuffer.AllocationMode.HEAP) {
            Object array = toSet.data().array();
            //data is assumed to have already been updated
            if(array == data)
                return;
            else {
                //copy the data over directly to the underlying array
                double[] d = (double[]) array;

                if(toSet.offset() == 0 && toSet.length() == data.length)
                    System.arraycopy(data,0,d,0,d.length);
                else {
                    int count = 0;
                    //need to do strided access with offset
                    for(int i = 0; i < data.length; i++) {
                        int dIndex = toSet.offset() + (i * toSet.majorStride());
                        d[dIndex] = data[count++];
                    }
                }
            }
        }
        else {
            //assumes the underlying data is in the right order
            DataBuffer underlyingData = toSet.data();
            if(data.length == toSet.length() && toSet.offset() == 0) {
                for(int i = 0; i < toSet.length(); i++) {
                    underlyingData.put(i,data[i]);
                }
            }
            else {
                int count = 0;
                //need to do strided access with offset
                for(int i = 0; i < data.length; i++) {
                    int dIndex = toSet.offset() + (i * toSet.majorStride());
                    underlyingData.put(dIndex,data[count++]);
                }
            }
        }


    }


}
