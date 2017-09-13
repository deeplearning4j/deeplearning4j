package org.deeplearning4j.nn.api.activations;

import lombok.AllArgsConstructor;
import org.deeplearning4j.nn.api.MaskState;
import org.nd4j.linalg.api.ndarray.INDArray;

@AllArgsConstructor
public class ActivationsTuple extends BaseActivations {

    private INDArray[] activations;
    private INDArray[] masks;
    private MaskState[] maskStates;

    @Override
    public int size() {
        return activations.length;
    }

    @Override
    public INDArray get(int idx) {
        assertIndex(idx);
        return activations[idx];
    }

    @Override
    public INDArray getMask(int idx) {
        assertIndex(idx);
        if(masks == null) return null;
        return masks[idx];
    }

    @Override
    public MaskState getMaskState(int idx) {
        assertIndex(idx);
        if(maskStates == null) return null;
        return maskStates[idx];
    }

    @Override
    public void set(int idx, INDArray activations) {
        assertIndex(idx);
        this.activations[idx] = activations;
    }

    @Override
    public void setMask(int idx, INDArray mask) {
        assertIndex(idx);
        if(masks == null) masks = new INDArray[activations.length];
        masks[idx] = mask;
    }

    @Override
    public void setMaskState(int idx, MaskState maskState) {
        assertIndex(idx);
        if(maskStates == null) maskStates = new MaskState[activations.length];
        maskStates[idx] = maskState;
    }

    @Override
    public INDArray[] getAsArray(){
        return activations;
    }

    @Override
    public INDArray[] getMaskAsArray(){
        return masks;
    }

    @Override
    public MaskState[] getMaskStateAsArray(){
        return maskStates;
    }
}
