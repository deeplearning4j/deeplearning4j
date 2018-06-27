package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.Data;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Map;

@Data
public abstract class SameDiffVertex extends GraphVertex {

    private SDVertexParams vertexParams;

    public abstract SDVariable defineVertex(SameDiff sameDiff, Map<String,SDVariable> layerInput, Map<String,SDVariable> paramTable);

    /**
     * Define the parameters - and inputs - for the network.
     * Use {@link SDVertexParams#addWeightParam(String, int...)} and
     * {@link SDVertexParams#addBiasParam(String, int[])}.
     * Note also you must define (and optionally name) the inputs to the vertex. This is required so that
     * DL4J knows how many inputs exists for the vertex.
     * @param params Object used to set parameters for this layer
     */
    public abstract void defineParametersAndInputs(SDVertexParams params);

    /**
     * Set the initial parameter values for this layer, if required
     * @param params Parameter arrays that may be initialized
     */
    public abstract void initializeParameters(Map<String,INDArray> params);

    public SDVertexParams getVertexParams(){
        if(vertexParams == null){
            vertexParams = new SDVertexParams();
            defineParametersAndInputs(vertexParams);
        }
        return vertexParams;
    }

    @Override
    public GraphVertex clone() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public int numParams(boolean backprop) {
        SDLayerParams params = getVertexParams();
        long count = 0;
        for(long[] l : params.getParamShapes().values()){
            count += ArrayUtil.prodLong(l);
        }
        return (int)count;
    }

    @Override
    public int minVertexInputs() {
        return 1;
    }

    @Override
    public int maxVertexInputs() {
        return -1;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx, INDArray paramsView, boolean initializeParams) {
        return null;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public MemoryReport getMemoryReport(InputType... inputTypes) {
        return null;
    }
}
