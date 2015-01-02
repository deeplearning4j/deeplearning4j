package org.deeplearning4j.scaleout.perform.models.word2vec;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by agibsonccc on 11/29/14.
 */
public class Word2VecResult implements Serializable {
    private Map<String,INDArray> syn0Change = new HashMap<>();
    private Map<String,INDArray> syn1Change = new HashMap<>();
    private Map<String,INDArray> negativeChange = new HashMap<>();



    public Word2VecResult() {}

    public Word2VecResult(Map<String, INDArray> syn0Change, Map<String, INDArray> syn1Change, Map<String, INDArray> negativeChange) {
        this.syn0Change = syn0Change;
        this.syn1Change = syn1Change;
        this.negativeChange = negativeChange;
    }

    public Map<String, INDArray> getSyn0Change() {
        return syn0Change;
    }

    public void setSyn0Change(Map<String, INDArray> syn0Change) {
        this.syn0Change = syn0Change;
    }

    public Map<String, INDArray> getSyn1Change() {
        return syn1Change;
    }

    public void setSyn1Change(Map<String, INDArray> syn1Change) {
        this.syn1Change = syn1Change;
    }

    public Map<String, INDArray> getNegativeChange() {
        return negativeChange;
    }

    public void setNegativeChange(Map<String, INDArray> negativeChange) {
        this.negativeChange = negativeChange;
    }
}
