package org.deeplearning4j.nn.api.activations;

import lombok.AllArgsConstructor;
import org.deeplearning4j.nn.api.MaskState;
import org.nd4j.linalg.api.ndarray.INDArray;

@AllArgsConstructor
public class ActivationsTriple extends BaseActivations {

    private INDArray activations1;
    private INDArray activations2;
    private INDArray activations3;
    private INDArray mask1;
    private INDArray mask2;
    private INDArray mask3;
    private MaskState maskState1;
    private MaskState maskState2;
    private MaskState maskState3;

    @Override
    public int size() {
        return 3;
    }

    @Override
    public INDArray get(int idx) {
        assertIndex(idx);
        switch (idx){
            case 0:
                return activations1;
            case 1:
                return activations2;
            case 2:
                return activations3;
            default:
                throw new RuntimeException();
        }
    }

    @Override
    public INDArray getMask(int idx) {
        assertIndex(idx);
        switch (idx){
            case 0:
                return mask1;
            case 1:
                return mask2;
            case 2:
                return mask3;
            default:
                throw new RuntimeException();
        }
    }

    @Override
    public MaskState getMaskState(int idx) {
        assertIndex(idx);
        switch (idx){
            case 0:
                return maskState1;
            case 1:
                return maskState2;
            case 2:
                return maskState3;
            default:
                throw new RuntimeException();
        }
    }

    @Override
    public void set(int idx, INDArray activations) {
        assertIndex(idx);
        switch (idx){
            case 0:
                activations1 = activations;
                return;
            case 1:
                activations2 = activations;
                return;
            case 2:
                activations3 = activations;
        }
    }

    @Override
    public void setMask(int idx, INDArray mask) {
        assertIndex(idx);
        switch (idx){
            case 0:
                mask1 = mask;
                return;
            case 1:
                mask2 = mask;
                return;
            case 2:
                mask3 = mask;
        }
    }

    @Override
    public void setMaskState(int idx, MaskState maskState) {
        assertIndex(idx);
        switch (idx){
            case 0:
                maskState1 = maskState;
                return;
            case 1:
                maskState2 = maskState;
                return;
            case 2:
                maskState3 = maskState;
        }
    }
}
