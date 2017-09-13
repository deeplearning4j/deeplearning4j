package org.deeplearning4j.nn.api.activations;

import org.deeplearning4j.nn.api.MaskState;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class BaseActivations implements Activations {

    protected void assertIndex(int idx){
        if(idx < 0 || idx >= size()){
            throw new IllegalArgumentException("Invalid index: cannot get/set index " + idx + " from activations of " +
                    "size " + size());
        }
    }

    @Override
    public void clear() {
        for( int i=0; i<size(); i++ ){
            set(i, null);
            setMask(i, null);
            setMaskState(i, null);
        }
    }

    @Override
    public INDArray[] getAsArray(){
        INDArray[] out = new INDArray[size()];
        for( int i=0; i<size(); i++ ){
            out[i] = get(i);
        }
        return out;
    }

    @Override
    public INDArray[] getMaskAsArray(){
        INDArray[] out = new INDArray[size()];
        for( int i=0; i<size(); i++ ){
            out[i] = getMask(i);
        }
        return out;
    }

    @Override
    public MaskState[] getMaskStateAsArray(){
        MaskState[] out = new MaskState[size()];
        for( int i=0; i<size(); i++ ){
            out[i] = getMaskState(i);
        }
        return out;
    }
}
