package org.deeplearning4j.nn.api.activations;

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
}
