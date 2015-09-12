package org.nd4j.linalg.api.iter;

import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.Serializable;

/**
 * Represents a cache linear index lookup
 * @author Adam Gibson
 */
public class LinearIndexLookup implements Serializable {
    private char ordering;
    private int[][] indexes;
    private int[] shape;
    private boolean[] exists;
    private int numIndexes;

    /**
     *
     * @param shape the shape of the linear index
     * @param ordering the ordering of the linear index
     */
    public LinearIndexLookup(int[] shape,char ordering) {
        this.shape = shape;
        this.ordering = ordering;
        numIndexes = ArrayUtil.prod(shape);
        indexes = new int[numIndexes][shape.length];
        exists = new boolean[numIndexes];
    }

    /**
     * Give back a sub
     * wrt the given linear index
     * @param index the index
     * @return the sub for the given index
     */
    public int[] lookup(int index) {
        if(exists[index]) {
            return indexes[index];
        }
        else {
            exists[index] = true;
            indexes[index] = ordering == 'c' ? Shape.ind2subC(shape,index,numIndexes) : Shape.ind2sub(shape,index,numIndexes);
            return indexes[index];
        }
    }


}
