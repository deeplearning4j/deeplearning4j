package org.deeplearning4j.nn.api.activations;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.MaskState;
import org.nd4j.linalg.api.ndarray.INDArray;

@AllArgsConstructor
public class ActivationsSingle extends BaseActivations {

    private INDArray activations;
    private INDArray mask;
    private MaskState maskState;

    @Override
    public int size() {
        return 1;
    }

    @Override
    public INDArray get(int idx) {
        assertIndex(idx);
        return activations;
    }

    @Override
    public INDArray getMask(int idx) {
        assertIndex(idx);
        return mask;
    }

    @Override
    public MaskState getMaskState(int idx) {
        assertIndex(idx);
        return maskState;
    }

    @Override
    public void set(int idx, INDArray activations) {
        assertIndex(idx);
        this.activations = activations;
    }

    @Override
    public void setMask(int idx, INDArray mask) {
        assertIndex(idx);
        this.maskState = maskState;
    }

    @Override
    public void setMaskState(int idx, MaskState maskState) {
        assertIndex(idx);
        this.maskState = maskState;
    }

    @Override
    public void clear() {
        activations = null;
        mask = null;
        maskState = null;
    }
}
