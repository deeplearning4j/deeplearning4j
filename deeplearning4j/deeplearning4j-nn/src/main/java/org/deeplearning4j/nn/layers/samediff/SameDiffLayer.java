package org.deeplearning4j.nn.layers.samediff;

import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.samediff.AbstractSameDiffLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.temp.ExternalErrorsFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.*;

public class SameDiffLayer extends AbstractLayer<AbstractSameDiffLayer> {

    public static final String INPUT_KEY = "input";

    protected SameDiff sameDiff;
    protected SDVariable outputVar;
    protected ExternalErrorsFunction fn;
    protected String outputKey;

    protected INDArray params;
    protected INDArray gradients;
    protected Map<String,INDArray> paramTable;
    protected Map<String,INDArray> gradTable;


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
        //TODO - properly support weight noise...
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        if(sameDiff == null){
            doInit();
        }

        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            sameDiff.clearExecutionCache();
            sameDiff.associateArrayWithVariable(input.dup(), sameDiff.getVariable(INPUT_KEY));
            for(String s : paramTable.keySet() ) {
                sameDiff.associateArrayWithVariable(paramTable.get(s), s);
            }

            sameDiff.exec();
            INDArray result = sameDiff.getArrForVarName(outputKey);
            return workspaceMgr.dup(ArrayType.ACTIVATIONS, result);
        }
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        Gradient g = new DefaultGradient();

        INDArray dLdIn;
        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()){
            sameDiff.clearExecutionCache();
            sameDiff.associateArrayWithVariable(input.dup(), sameDiff.getVariable(INPUT_KEY));
            fn.updateVariable(outputVar.getVarName(), epsilon.dup());

            for(String s : paramTable.keySet() ){
                //TODO this should only be necessary, in theory, once!
                sameDiff.associateArrayWithVariable(paramTable.get(s), s);
            }

            sameDiff.execBackwards();
            for(String s : paramTable.keySet() ){
                INDArray sdGrad = sameDiff.grad(s).getArr();
                INDArray dl4jGrad = gradTable.get(s);
                dl4jGrad.assign(sdGrad);                                            //TODO OPTIMIZE THIS
                g.gradientForVariable().put(s, dl4jGrad);
            }

            dLdIn = sameDiff.grad(INPUT_KEY).getArr();
        }

        return new Pair<>(g, workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, dLdIn));   //TODO OPTIMIZE THIS
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
    public int numParams(){
        return params == null ? 0 : (int)params.length();
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
        this.gradTable = layerConf().initializer().getGradientsFromFlattened(conf(), gradients);
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
        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer bl = (org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer) layerConf();
            sameDiff = SameDiff.create();
            Map<String, INDArray> p = paramTable();

            val inputShape = input.shape().clone();
//        inputShape[0] = -1;                                       //TODO THIS DOESN'T ENABLE VARIABLE SIZE MINIBATCHES
            SDVariable inputVar = sameDiff.var(INPUT_KEY, inputShape);
            Map<String, long[]> paramShapes = layerConf().getLayerParams().getParamShapes();
            Map<String, SDVariable> params = new LinkedHashMap<>();
            for (String s : paramShapes.keySet()) {
                val ps = paramShapes.get(s);
                SDVariable v = sameDiff.var(s, ps);
                params.put(s, v);
            }
            SDVariable layerOutput = bl.defineLayer(sameDiff, inputVar, params);
            Preconditions.checkNotNull(layerOutput, "Invalid output: layer output is null");
            outputVar = layerOutput;

            for (Map.Entry<String, INDArray> e : p.entrySet()) {
                sameDiff.associateArrayWithVariable(e.getValue(), sameDiff.getVariable(e.getKey()));
            }

            //Define the function for external errors:
            fn = sameDiff.f().externalErrors(layerOutput);
            fn.outputVariable();

            this.outputKey = outputVar.getVarName();
        }
    }
}
