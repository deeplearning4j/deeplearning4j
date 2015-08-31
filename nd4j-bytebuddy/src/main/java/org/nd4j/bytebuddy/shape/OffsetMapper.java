package org.nd4j.bytebuddy.shape;

/**
 * Maps a given shape,stride and index
 * sub on to a linear index relative
 * to the given base offset
 *
 * @author Adam Gibson
 */
public interface OffsetMapper {

    /**
     * Get an offset for retrieval
     * from a data buffer
     * based on the given
     * shape stride and given indices
     * @param baseOffset the offset to start from
     * @param shape the shape of the array
     * @param stride the stride of the array
     * @param indices the indices to iterate over
     * @return the double at the specified index
     */
    int getOffset(int baseOffset,int[] shape,int[] stride,int[] indices);
}
