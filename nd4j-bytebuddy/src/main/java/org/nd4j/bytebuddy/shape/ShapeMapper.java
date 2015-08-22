package org.nd4j.bytebuddy.shape;

import net.bytebuddy.implementation.Implementation;
import org.nd4j.bytebuddy.arrays.assign.relative.RelativeAssignImplementation;
import org.nd4j.bytebuddy.arrays.create.noreturn.IntArrayCreation;
import org.nd4j.bytebuddy.dup.DuplicateImplementation;

/**
 * @author Adam Gibson
 */
public class ShapeMapper {
    /**
     * Algorithm does the following:
     * duplicate the top of the stack
     * the top of the stack represents the target int array
     *
     * assign the given dimension (putting an int on the stack represents the index)
     * @param shape
     * @param linearIndex
     * @param denom
     * @param dimension
     * @return
     */
    public static Implementation getImplementationForIndex(int[] shape,int linearIndex,int denom,int dimension) {
        return new Implementation.Compound(
                new DuplicateImplementation(),
                new RelativeAssignImplementation(dimension, 5)

        );
    }

    public static Implementation getImplementation(int[] shape,int index,int numIndices) {
        return new Implementation.Compound(
                new IntArrayCreation(shape.length)
        );
    }

}
