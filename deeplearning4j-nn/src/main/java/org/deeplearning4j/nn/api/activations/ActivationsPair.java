package org.deeplearning4j.nn.api.activations;

import lombok.AllArgsConstructor;
import org.deeplearning4j.nn.api.MaskState;
import org.nd4j.linalg.api.ndarray.INDArray;

@AllArgsConstructor
public class ActivationsPair extends BaseActivations {

    private INDArray activations1;
    private INDArray activations2;
    private INDArray mask1;
    private INDArray mask2;
    private MaskState maskState1;
    private MaskState maskState2;

    @Override
    public int size() {
        return 2;
    }

    @Override
    public INDArray get(int idx) {
        assertIndex(idx);
        return (idx == 0 ? activations1 : activations2);
    }

    @Override
    public INDArray getMask(int idx) {
        assertIndex(idx);
        return (idx == 0 ? mask1 : mask2);
    }

    @Override
    public MaskState getMaskState(int idx) {
        assertIndex(idx);
        return (idx == 0 ? maskState1 : maskState2);
    }

    @Override
    public void set(int idx, INDArray activations) {
        assertIndex(idx);
        if(idx == 0){
            activations1 = activations;
        } else {
            activations2 = activations;
        }
    }

    @Override
    public void setMask(int idx, INDArray mask) {
        assertIndex(idx);
        if(idx == 0){
            mask1 = mask;
        } else {
            mask2 = mask;
        }
    }

    @Override
    public void setMaskState(int idx, MaskState maskState) {
        assertIndex(idx);
        if(idx == 0){
            maskState1 = maskState;
        } else {
            maskState2 = maskState;
        }
    }

    @Override
    public Activations cloneShallow() {
        return new ActivationsPair(activations1, activations2, mask1, mask2, maskState1, maskState2);
    }
}
