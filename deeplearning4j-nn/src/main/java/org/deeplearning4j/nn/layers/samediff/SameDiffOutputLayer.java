package org.deeplearning4j.nn.layers.samediff;

import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffOutputLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class SameDiffOutputLayer extends SameDiffLayer implements IOutputLayer {

    public static final String LABEL_KEY = "label";



    public SameDiffOutputLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public void setLabels(INDArray labels) {

    }

    @Override
    public INDArray getLabels() {
        return null;
    }

    @Override
    public double computeScore(double fullNetworkL1, double fullNetworkL2, boolean training) {
        return 0;
    }

    @Override
    public INDArray computeScoreForExamples(double fullNetworkL1, double fullNetworkL2) {
        return null;
    }

    @Override
    public double f1Score(DataSet data) {
        throw new UnsupportedOperationException();
    }

    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int numLabels() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fit(DataSetIterator iter) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int[] predict(INDArray examples) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> predict(DataSet dataSet) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray labelProbabilities(INDArray examples) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fit(INDArray examples, INDArray labels) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fit(DataSet data) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fit(INDArray examples, int[] labels) {
        throw new UnsupportedOperationException();
    }

    protected void doInit(){
        BaseSameDiffOutputLayer ol = ((BaseSameDiffOutputLayer)layerConf());

        sameDiff = SameDiff.create();
        Map<String,INDArray > p = paramTable();

        int[] inputShape = input.shape().clone();
        int[] labelShape = ol.labelShape();
//        inputShape[0] = -1;                                       //TODO THIS DOESN'T ENABLE VARIABLE SIZE MINIBATCHES
        SDVariable inputVar = sameDiff.var(INPUT_KEY, inputShape);
        SDVariable labelVar = sameDiff.var(LABEL_KEY, labelShape);
        Map<String,int[]> paramShapes = layerConf().paramShapes();
        Map<String,SDVariable> params = new LinkedHashMap<>();
        for(String s : layerConf().paramKeys()){
            int[] ps = paramShapes.get(s);
            SDVariable v = sameDiff.var(s, ps);
            params.put(s, v);
        }
        List<String> outputKeys = ol.defineLayer(sameDiff, inputVar, labelVar, params);
        if(outputKeys == null || outputKeys.size() != 1){
            throw new IllegalStateException("Invalid output keys: " + outputKeys);
        }

        for(Map.Entry<String,INDArray> e : p.entrySet()){
            sameDiff.associateArrayWithVariable(e.getValue(), sameDiff.getVariable(e.getKey()));
        }

        this.outputKeys = outputKeys;
    }
}
