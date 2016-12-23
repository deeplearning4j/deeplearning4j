package org.deeplearning4j.keras;

import org.nd4j.linalg.api.ndarray.INDArray;

public class RecursiveCopier {

    private final INDArray output;
    private final float[] input;
    private final int[] shape;
    private int inputIdx = 0;

    public RecursiveCopier(INDArray output, float[] input, int[] shape) {
        this.output = output;
        this.input = input;
        this.shape = shape;
    }

    public void copy() {
        copyRecursive(0, new int[shape.length]);
    }

    private void copyRecursive(int shapeIdx, int[] indexes) {
        if (shape.length == shapeIdx) {
            output.putScalar(indexes, input[inputIdx++]);
        } else {
            for (int i = 0; i < shape[shapeIdx]; i++) {
                indexes[shapeIdx] = i;
                copyRecursive(shapeIdx + 1, indexes);
            }
        }
    }

}
