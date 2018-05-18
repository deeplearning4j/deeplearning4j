package org.deeplearning4j.nn.layers.samediff;

import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.samediff.AbstractSameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.*;

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
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        if(sameDiff == null){
            doInit();
        }

        sameDiff.associateArrayWithVariable(input, sameDiff.getVariable(INPUT_KEY));

        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            INDArray result = sameDiff.execAndEndResult();
            return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, result);
        }
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Fitting DL4J SameDiff layers via backpropagation is not yet supported");

        /*
        assertInputSet(true);
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
        */
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
        return paramTable.get(param);
    }

    @Override
    public void setParam(String key, INDArray val) {
        if(!paramTable.containsKey(key)){
            throw new IllegalArgumentException("Cannot set parameter, invalid/unknown parameter key: " + key);
        }
        INDArray current = paramTable.get(key);
        if(!Arrays.equals(current.shape(), val.shape())){
            throw new IllegalArgumentException("Cannot set parameter \"" + key + "\", invalid shape: parameter array has shape "
                    + Arrays.toString(current.shape()) + ", trying to set parameter of shape " + Arrays.toString(val.shape()));
        }
    }

    @Override
    public void setParams(INDArray params) {
        if (params != null) {
            throw new UnsupportedOperationException("Not supported");
        }
    }

    protected void setParams(INDArray params, char order) {
        setParams(params);
    }

    @Override
    public void setParamsViewArray(INDArray params) {
        this.params = params;
    }

    @Override
    public INDArray getGradientsViewArray() {
        return gradients;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        this.gradients = gradients;
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        if(this.paramTable == null){
            this.paramTable = paramTable;
        } else {
            for (Map.Entry<String, INDArray> e : paramTable.entrySet()) {
                setParam(e.getKey(), e.getValue());
            }
        }
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

        val inputShape = input.shape().clone();
//        inputShape[0] = -1;                                       //TODO THIS DOESN'T ENABLE VARIABLE SIZE MINIBATCHES
        SDVariable inputVar = sameDiff.var(INPUT_KEY, inputShape);
        Map<String,int[]> paramShapes = layerConf().getLayerParams().getParamShapes();
        Map<String,SDVariable> params = new LinkedHashMap<>();
        for(String s : paramShapes.keySet()){
            int[] ps = paramShapes.get(s);
            SDVariable v = sameDiff.var(s, ps);
            params.put(s, v);
        }
        List<SDVariable> layerOutputs = bl.defineLayer(sameDiff, inputVar, params);
        if(layerOutputs == null || layerOutputs.size() != 1){
            throw new IllegalStateException("Invalid outputs: " + layerOutputs);
        }

        for(Map.Entry<String,INDArray> e : p.entrySet()){
            sameDiff.associateArrayWithVariable(e.getValue(), sameDiff.getVariable(e.getKey()));
        }

        this.outputKeys = new ArrayList<>();
        for(SDVariable sdv : layerOutputs){
            outputKeys.add(sdv.getVarName());
        }
    }
}
