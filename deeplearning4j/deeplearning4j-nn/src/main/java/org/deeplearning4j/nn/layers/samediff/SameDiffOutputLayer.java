package org.deeplearning4j.nn.layers.samediff;

import lombok.Getter;
import lombok.Setter;
import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.temp.ExternalErrorsFunction;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;

public class SameDiffOutputLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.samediff.SameDiffOutputLayer>
    implements IOutputLayer {

    public static final String INPUT_KEY = "input";
    public static final String LABELS_KEY = "labels";

    protected SameDiff sameDiff;
    protected SDVariable outputVar;
    protected String outputKey;

    @Getter @Setter
    protected INDArray labels;

    protected INDArray params;
    protected INDArray gradients;
    protected Map<String,INDArray> paramTable;
    protected Map<String,INDArray> gradTable;


    public SameDiffOutputLayer(NeuralNetConfiguration conf){
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
        return activateHelper(true, workspaceMgr);
    }

    private INDArray activateHelper(boolean activations, LayerWorkspaceMgr workspaceMgr){
        assertInputSet(false);

        //Check where the output occors. If it's a simple loss layer (no params) this could
        // just be the input!
        if(activations && INPUT_KEY.equals(layerConf().activationsVertexName())){
            return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, input);
        }

        if(sameDiff == null){
            doInit();
        }

        //TODO optimize
        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            sameDiff.clearExecutionCache();
            sameDiff.associateArrayWithVariable(input.dup(), sameDiff.getVariable(INPUT_KEY));
            if(layerConf().labelsRequired()) {
                sameDiff.associateArrayWithVariable(labels.dup(), sameDiff.getVariable(LABELS_KEY));
            }
            for(String s : paramTable.keySet() ) {
                sameDiff.associateArrayWithVariable(paramTable.get(s), s);
            }

            INDArray score = sameDiff.execAndEndResult();
            if(activations) {
                INDArray result = sameDiff.getArrForVarName(layerConf().activationsVertexName());
                Preconditions.checkNotNull(result, "Activations (result) array for variable \"%s\" was" +
                        "null - error during execution or this variable (as defined by method activationsVertexName()) " +
                        "does not exist", layerConf().activationsVertexName());
                return workspaceMgr.dup(ArrayType.ACTIVATIONS, result);
            } else {
                return score;
            }
        }
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        Preconditions.checkState(!layerConf().labelsRequired() || labels != null, "Cannot execute backprop: Labels are not set. " +
                "If labels are not required for this SameDiff output layer, override SameDiffOutputLayer.labelsRequired()" +
                " to return false instead");

        Gradient g = new DefaultGradient();

        INDArray dLdIn;
        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()){
            sameDiff.clearExecutionCache();
            sameDiff.associateArrayWithVariable(input.dup(), sameDiff.getVariable(INPUT_KEY));
            if(layerConf().labelsRequired()) {
                sameDiff.associateArrayWithVariable(labels.dup(), sameDiff.getVariable(LABELS_KEY));
            }

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
            org.deeplearning4j.nn.conf.layers.samediff.SameDiffOutputLayer bl = layerConf();
            sameDiff = SameDiff.create();
            Map<String, INDArray> p = paramTable();

            val inputShape = input.shape().clone();
//        inputShape[0] = -1;                                       //TODO THIS DOESN'T ENABLE VARIABLE SIZE MINIBATCHES
            SDVariable inputVar = sameDiff.var(INPUT_KEY, inputShape);
            SDVariable labelVar = null;
            if(layerConf().labelsRequired()){
                long[] labelShape = labels.shape().clone();
                labelVar = sameDiff.var(LABELS_KEY, labelShape);
            }
            Map<String, long[]> paramShapes = layerConf().getLayerParams().getParamShapes();
            Map<String, SDVariable> params = new LinkedHashMap<>();
            for (String s : paramShapes.keySet()) {
                val ps = paramShapes.get(s);
                SDVariable v = sameDiff.var(s, ps);
                params.put(s, v);
            }
            SDVariable layerOutput = bl.defineLayer(sameDiff, inputVar, labelVar, params);
            Preconditions.checkNotNull(layerOutput, "Invalid output: layer output is null");
            outputVar = layerOutput;

            for (Map.Entry<String, INDArray> e : p.entrySet()) {
                sameDiff.associateArrayWithVariable(e.getValue(), sameDiff.getVariable(e.getKey()));
            }

            this.outputKey = layerOutput.getVarName();

//        sameDiff.createGradFunction();
        }
    }

    @Override
    public boolean needsLabels() {
        return layerConf().labelsRequired();
    }

    @Override
    public double computeScore(double fullNetworkL1, double fullNetworkL2, boolean training, LayerWorkspaceMgr workspaceMgr) {
        return activateHelper(false, workspaceMgr).getDouble(0) + fullNetworkL1 + fullNetworkL1;
    }

    @Override
    public INDArray computeScoreForExamples(double fullNetworkL1, double fullNetworkL2, LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public double f1Score(DataSet data) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public double f1Score(INDArray examples, INDArray labels) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public int numLabels() {
        return 0;
    }

    @Override
    public void fit(DataSetIterator iter) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public int[] predict(INDArray examples) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public List<String> predict(DataSet dataSet) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public INDArray labelProbabilities(INDArray examples) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void fit(INDArray examples, INDArray labels) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void fit(DataSet data) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void fit(INDArray examples, int[] labels) {
        throw new UnsupportedOperationException("Not supported");
    }
}
