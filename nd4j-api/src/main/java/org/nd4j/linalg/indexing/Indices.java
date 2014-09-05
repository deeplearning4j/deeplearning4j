package org.nd4j.linalg.indexing;

import org.nd4j.linalg.factory.Nd4j;

/**
 * Indexing util.
 * @author Adam Gibson
 */
public class Indices {

    /**
     * The offsets (begin index) for each index
     * @param indices the indices
     * @return the offsets for the given set of indices
     *
     */
    public static int[] offsets(NDArrayIndex...indices) {
        int[] ret = new int[indices.length];
        for(int i = 0; i < indices.length; i++) {
            int offset = indices[i].offset();
            if(offset == 0 && i > 0 && i < indices.length - 1)
                ret[i] = 1;
            else
                ret[i] = indices[i].offset();
        }
        return ret;
    }


    /**
     * Fill in the missing indices to be the
     * same length as the original shape.
     *
     * Think of this as what fills in the indices for numpy or matlab:
     * Given a which is (4,3,2) in numpy:
     *
     * a[1:3] is filled in by the rest
     * to give back the full slice
     *
     * This algorithm fills in that delta
     *
     * @param shape the original shape
     * @param indexes the indexes to start from
     * @return the filled in indices
     */
    public static NDArrayIndex[] fillIn(int[] shape,NDArrayIndex...indexes) {
        if(shape.length == indexes.length)
            return indexes;

        NDArrayIndex[] newIndexes = new NDArrayIndex[shape.length];
        System.arraycopy(indexes,0,newIndexes,0,indexes.length);

        for(int i = indexes.length; i < shape.length; i++) {
            newIndexes[i] = NDArrayIndex.interval(0,shape[i]);
        }
        return newIndexes;

    }

    /**
     * Prunes indices of greater length than the shape
     * and fills in missing indices if there are any
     * @param originalShape the original shape to adjust to
     * @param indexes the indexes to adjust
     * @return the  adjusted indices
     */
    public static NDArrayIndex[] adjustIndices(int[] originalShape,NDArrayIndex...indexes) {
        if(indexes.length < originalShape.length)
            indexes = fillIn(originalShape,indexes);
       if(indexes.length > originalShape.length) {
           NDArrayIndex[] ret = new NDArrayIndex[originalShape.length];
           System.arraycopy(indexes,0,ret,0,originalShape.length);
           return ret;
       }

        if(indexes.length == originalShape.length)
            return indexes;
        for(int i = 0; i < indexes.length; i++) {
            if(indexes[i].end() >= originalShape[i])
                indexes[i] = NDArrayIndex.interval(0,originalShape[i] - 1);
        }

        return indexes;
    }


    /**
     * Calculate the strides based on the given indices
     * @param ordering the ordering to calculate strides for
     * @param indexes the indices to calculate stride for
     * @return the strides for the given indices
     */
    public static int[] strides(char ordering,NDArrayIndex...indexes) {
        return Nd4j.getStrides(shape(indexes), ordering);
    }

    /**
     * Calculate the shape for the given set of indices.
     *
     * The shape is defined as (for each dimension)
     * the difference between the end index + 1 and
     * the begin index
     * @param indices the indices to calculate the shape for
     * @return the shape for the given indices
     */
    public static int[] shape(NDArrayIndex...indices) {
        int[] ret = new int[indices.length];
        for(int i = 0; i < ret.length; i++) {
            int[] currIndices = indices[i].indices();

            int end  = currIndices[currIndices.length - 1] + 1;
            int begin = currIndices[0];
            ret[i] = Math.abs(end - begin);
        }

        return ret;
    }



    /**
     * Calculate the shape for the given set of indices.
     *
     * The shape is defined as (for each dimension)
     * the difference between the end index + 1 and
     * the begin index
     *
     * If specified, this will check for whether any of the indices are >= to end - 1
     * and if so, prune it down
     *
     * @param shape the original shape
     * @param indices the indices to calculate the shape for
     * @return the shape for the given indices
     */
    public static int[] shape(int[] shape,NDArrayIndex...indices) {
        if(indices.length > shape.length)
            return shape;

        int[] ret = new int[indices.length];
        for(int i = 0; i < ret.length; i++) {
            int[] currIndices = indices[i].indices();

            int end  = currIndices[currIndices.length - 1] + 1;
            if(end > shape[i])
                end = shape[i] - 1;
            int begin = currIndices[0];
            ret[i] = Math.abs(end - begin);
        }

        return ret;
    }






}
