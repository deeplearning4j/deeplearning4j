package org.deeplearning4j.keras;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Copies data from input (DataBuffer) to output (INDArray) preserving the shape of the input data
 *
 * @author pkoperek@gmail.com
 */
public class RecursiveCopier {

    private final INDArray output;
    private final DataBuffer input;
    private final int[] shape;
    private int inputIdx = 0;

    public RecursiveCopier(INDArray output, DataBuffer input, int[] shape) {
        this.output = output;
        this.input = input;
        this.shape = shape;
    }

    public void copy() {
        copyRecursive(0, new int[shape.length]);
    }

    private void copyRecursive(int shapeIdx, int[] indexes) {
        if (shape.length == shapeIdx) {
            output.putScalar(indexes, input.getFloat(inputIdx++));
        } else {
            for (int i = 0; i < shape[shapeIdx]; i++) {
                indexes[shapeIdx] = i;
                copyRecursive(shapeIdx + 1, indexes);
            }
        }
    }

}
