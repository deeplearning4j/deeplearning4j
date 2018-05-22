package org.deeplearning4j.nn.modelimport.keras.preprocessors;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.preprocessor.BaseInputPreProcessor;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.shape.Shape;

/**
 * Preprocessor to flatten input of RNN type
 *
 * @author Max Pumperla
 */
@Slf4j
public class KerasFlattenRnnPreprocessor extends BaseInputPreProcessor {

    private long tsLength;
    private long depth;

    public KerasFlattenRnnPreprocessor(long depth, long tsLength) {
        super();
        this.tsLength = Math.abs(tsLength);
        this.depth = depth;
    }

    @Override
    public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        INDArray output = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'c');
        return output.reshape(input.size(0), depth * tsLength);
    }

    @Override
    public INDArray backprop(INDArray epsilons, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
        return workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, epsilons, 'c').reshape(miniBatchSize, depth, tsLength);
    }

    @Override
    public KerasFlattenRnnPreprocessor clone() {
        return (KerasFlattenRnnPreprocessor) super.clone();
    }

    @Override
    public InputType getOutputType(InputType inputType) throws InvalidInputTypeException {

        return InputType.feedForward(depth * tsLength);

    }
}
