package org.deeplearning4j.nn.api.gradients;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

@AllArgsConstructor
public abstract class BaseGradients implements Gradients {

    @Getter @Setter
    protected Gradient parameterGradients;

    @Override
    public void clear() {
        for( int i=0; i<size(); i++ ){
            setActivationGrad(i, null);
        }
        parameterGradients = null;
    }

    protected void assertIndex(int idx){
        if(idx < 0 || idx >= size()){
            throw new IllegalArgumentException("Invalid index: cannot get/set index " + idx + " from activation" +
                    " gradients of size " + size());
        }
    }

    @Override
    public INDArray[] getActivationGradAsArray(){
        INDArray[] out = new INDArray[size()];
        for( int i=0; i<size(); i++ ){
            out[i] = getActivationGrad(i);
        }
        return out;
    }
}
