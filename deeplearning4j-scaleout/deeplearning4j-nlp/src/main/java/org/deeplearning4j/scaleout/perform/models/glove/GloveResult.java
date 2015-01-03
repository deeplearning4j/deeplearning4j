package org.deeplearning4j.scaleout.perform.models.glove;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Result for glove
 * @author Adam Gibson
 */
public class GloveResult implements Serializable {
    private Map<String,INDArray> syn0Change = new HashMap<>();



    public GloveResult() {}

    public GloveResult(Map<String, INDArray> syn0Change) {
        this.syn0Change = syn0Change;

    }

    public Map<String, INDArray> getSyn0Change() {
        return syn0Change;
    }

    public void setSyn0Change(Map<String, INDArray> syn0Change) {
        this.syn0Change = syn0Change;
    }


}
