package org.deeplearning4j.nn.layers.samediff;

import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.layers.samediff.SDVertexParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffVertex;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.params.SameDiffParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.temp.ExternalErrorsFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

public class SameDiffGraphVertex extends BaseGraphVertex {

    protected SameDiffVertex config;
    protected SameDiff sameDiff;
    protected SDVariable outputVar;
    protected ExternalErrorsFunction fn;
    protected String outputKey;
    protected Map<String,SDVariable> inputVars;

    protected INDArray params;
    protected INDArray gradients;
    protected Map<String,INDArray> paramTable;
    protected Map<String,INDArray> gradTable;

    public SameDiffGraphVertex(SameDiffVertex config, ComputationGraph graph, String name, int vertexIndex,
                                  INDArray paramsView, boolean initParams) {
        super(graph, name, vertexIndex, null, null);
        this.config = config;
        SDVertexParams vp = config.getVertexParams();
        paramTable = SameDiffParamInitializer.getInstance().subsetAndReshape(vp.getParameterKeys(),
                vp.getParamShapes(), paramsView, null, config);
        if(initParams){
            config.initializeParameters(paramTable);
        }
    }

    @Override
    public String toString() {
        return null;
    }

    @Override
    public boolean hasLayer() {
        return false;
    }

    @Override
    public Layer getLayer() {
        return null;
    }

    @Override
    public INDArray doForward(boolean training, LayerWorkspaceMgr workspaceMgr) {
        if(sameDiff == null){
            doInit();
        }

        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            sameDiff.clearExecutionCache();
            for(int i=0; i<inputs.length; i++ ){
                String name = config.getVertexParams().getInputs().get(i);
                sameDiff.associateArrayWithVariable(inputs[i].dup(), sameDiff.getVariable(name));
            }
            if(paramTable != null && paramTable.size() > 0) {
                for (String s : paramTable.keySet()) {
                    sameDiff.associateArrayWithVariable(paramTable.get(s), s);
                }
            }
            sameDiff.exec();
            INDArray result = sameDiff.getArrForVarName(outputKey);
            return workspaceMgr.dup(ArrayType.ACTIVATIONS, result);
        }
    }

    @Override
    public Pair<Gradient, INDArray[]> doBackward(boolean tbptt, LayerWorkspaceMgr workspaceMgr) {
        Gradient g = new DefaultGradient();

        INDArray[] dLdIns;
        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()){
            sameDiff.clearExecutionCache();
            //Set inputs
            for(int i=0; i<inputs.length; i++ ){
                String name = config.getVertexParams().getInputs().get(i);
                sameDiff.associateArrayWithVariable(inputs[i].dup(), sameDiff.getVariable(name));
            }
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

            dLdIns = new INDArray[inputs.length];
            for(int i=0; i<inputs.length; i++ ){
                String name = config.getVertexParams().getInputs().get(i);
                dLdIns[i] = sameDiff.grad(name).getArr();
            }
        }

        //TODO optimize
        for( int i=0; i<dLdIns.length; i++ ){
            dLdIns[i] = workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, dLdIns[i]);
        }

        return new Pair<>(g, dLdIns);
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray backpropGradientsViewArray) {
        SDVertexParams vp = config.getVertexParams();
        gradTable = SameDiffParamInitializer.getInstance().subsetAndReshape(vp.getParameterKeys(),
                vp.getParamShapes(), backpropGradientsViewArray, null, config);
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        throw new UnsupportedOperationException("Not yet supported");
    }


    protected void doInit(){
        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            sameDiff = SameDiff.create();

            inputVars = new HashMap<>();
            int i=0;
            for(String s : config.getVertexParams().getInputs()){
                val inputShape = inputs[i++].shape().clone();
                SDVariable inputVar = sameDiff.var(s, inputShape);
                inputVars.put(s, inputVar);
            }

            Map<String, long[]> paramShapes = config.getVertexParams().getParamShapes();
            Map<String, SDVariable> params = new LinkedHashMap<>();
            for (String s : paramShapes.keySet()) {
                val ps = paramShapes.get(s);
                SDVariable v = sameDiff.var(s, ps);
                params.put(s, v);
            }
            SDVariable layerOutput = config.defineVertex(sameDiff, inputVars, params);
            Preconditions.checkNotNull(layerOutput, "Invalid output: layer output is null");
            outputVar = layerOutput;

            for (Map.Entry<String, INDArray> e : paramTable.entrySet()) {
                sameDiff.associateArrayWithVariable(e.getValue(), sameDiff.getVariable(e.getKey()));
            }

            //Define the function for external errors:
            fn = sameDiff.f().externalErrors(layerOutput);
            fn.outputVariable();

            this.outputKey = outputVar.getVarName();
        }
    }
}


