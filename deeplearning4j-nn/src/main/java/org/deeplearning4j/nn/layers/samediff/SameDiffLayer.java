package org.deeplearning4j.nn.layers.samediff;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.samediff.AbstractSameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class SameDiffLayer extends AbstractLayer<AbstractSameDiffLayer> {

    public static final String INPUT_KEY = "input";

    protected SameDiff sameDiff;
    protected List<String> outputKeys;

    protected INDArray params;
    protected INDArray gradients;
    protected Map<String,INDArray> paramTable;


    public SameDiffLayer(NeuralNetConfiguration conf){
        super(conf);
    }



    @Override
    public Layer clone() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {
        //TODO - properly support noise weight...
    }

    @Override
    public INDArray activate(boolean training) {
        if(sameDiff == null){
            doInit();
        }

        //Build map:
//        Map<String, INDArray> map = new HashMap<>(paramTable());
//        map.put(INPUT_KEY, input);

        sameDiff.associateArrayWithVariable(input, sameDiff.getVariable(INPUT_KEY));

        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            INDArray result = sameDiff.execAndEndResult();
            return result;
        }
    }

    @Override
    public INDArray preOutput(boolean training) {
        return activate(training);
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        Gradient g = new DefaultGradient();

        INDArray dLdIn;
        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()){
            sameDiff.execBackwards();
            for(String s : layerConf().paramKeys() ){
                INDArray pg = sameDiff.grad(s).getArr();
                g.gradientForVariable().put(s, pg);
            }

            dLdIn = sameDiff.grad(INPUT_KEY).getArr();
        }

        return new Pair<>(g, dLdIn);
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        double l2Sum = 0.0;
        for (Map.Entry<String, INDArray> entry : paramTable().entrySet()) {
            double l2 = conf.getL2ByParam(entry.getKey());
            if (l2 > 0) {
                double norm2 = getParam(entry.getKey()).norm2Number().doubleValue();
                l2Sum += 0.5 * l2 * norm2 * norm2;
            }
        }

        return l2Sum;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        double l1Sum = 0.0;
        for (Map.Entry<String, INDArray> entry : paramTable().entrySet()) {
            double l1 = conf.getL1ByParam(entry.getKey());
            if (l1 > 0) {
                double norm1 = getParam(entry.getKey()).norm1Number().doubleValue();
                l1Sum += l1 * norm1;
            }
        }

        return l1Sum;
    }

    /**Returns the parameters of the neural network as a flattened row vector
     * @return the parameters of the neural network
     */
    @Override
    public INDArray params() {
        return params;
    }

    @Override
    public INDArray getParam(String param) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void setParam(String key, INDArray val) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void setParams(INDArray params) {
        if (params != null) {
            throw new UnsupportedOperationException("Not supported");
        }
    }

    protected void setParams(INDArray params, char order) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void setParamsViewArray(INDArray params) {
        this.params = params;
    }

    @Override
    public INDArray getGradientsViewArray() {
        return params;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        this.gradients = gradients;
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        this.paramTable = paramTable;
    }

    @Override
    public Map<String, INDArray> paramTable() {
        return paramTable(false);
    }

    @Override
    public Map<String, INDArray> paramTable(boolean backpropParamsOnly) {
        return paramTable;
    }

    protected void doInit(){
        BaseSameDiffLayer bl = (BaseSameDiffLayer)layerConf();
        sameDiff = SameDiff.create();
        Map<String,INDArray > p = paramTable();

        int[] inputShape = input.shape().clone();
//        inputShape[0] = -1;                                       //TODO THIS DOESN'T ENABLE VARIABLE SIZE MINIBATCHES
        SDVariable inputVar = sameDiff.var(INPUT_KEY, inputShape);
        Map<String,int[]> paramShapes = layerConf().paramShapes();
        Map<String,SDVariable> params = new LinkedHashMap<>();
        for(String s : layerConf().paramKeys()){
            int[] ps = paramShapes.get(s);
            SDVariable v = sameDiff.var(s, ps);
            params.put(s, v);
        }
        List<String> outputKeys = bl.defineLayer(sameDiff, inputVar, params);
        if(outputKeys == null || outputKeys.size() != 1){
            throw new IllegalStateException("Invalid output keys: " + outputKeys);
        }

        for(Map.Entry<String,INDArray> e : p.entrySet()){
            sameDiff.associateArrayWithVariable(e.getValue(), sameDiff.getVariable(e.getKey()));
        }

        this.outputKeys = outputKeys;
    }
}
