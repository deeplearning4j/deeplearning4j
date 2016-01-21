package org.deeplearning4j.nn.graph.vertex;

import lombok.Data;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Arrays;

/** A graph vertex is a vertex in the computation graph. It may contain either a Layer, or a GraphNode
 * The purpose of  the GraphVertex class is as follows:
 * 1. To track the (local) network connection structure: i.e., it knows about the nodes on the input and output sides
 * 2. To store intermediate results (activations and epsilons)
 * 3. To allow forward pass and backward pass to be conducted, once the intermediate results are
 *
 */
@Data
public class LayerVertex extends BaseGraphVertex {

    private Layer layer;
    private InputPreProcessor layerPreProcessor;

    /** Create a network input vertex: */
    public LayerVertex(ComputationGraph graph, String name, int vertexIndex){
        this(graph, name, vertexIndex, null, null, null, null);
    }

    public LayerVertex(ComputationGraph graph, String name, int vertexIndex, VertexIndices[] inputVertices, VertexIndices[] outputVertices,
                        Layer layer, InputPreProcessor layerPreProcessor){
        super(graph,name,vertexIndex,inputVertices,outputVertices);
        this.graph = graph;
        this.vertexName = name;
        this.vertexIndex = vertexIndex;
        this.inputVertices = inputVertices;
        this.outputVertices = outputVertices;
        this.layer = layer;
        this.layerPreProcessor = layerPreProcessor;

        this.inputs = new INDArray[(inputVertices != null ? inputVertices.length : 0)];
        this.epsilons = new INDArray[(outputVertices != null ? outputVertices.length : 0)];
    }

    @Override
    public boolean hasLayer(){
        return true;
    }

    @Override
    public boolean isOutputVertex(){
        return layer instanceof BaseOutputLayer;
    }

    @Override
    public Layer getLayer(){
        return layer;
    }

    @Override
    public INDArray doForward(boolean training){
        if(!canDoForward()) throw new IllegalStateException("Cannot do forward pass: all inputs not set");

        if(inputs.length > 1) throw new UnsupportedOperationException("Not implemented");   //TODO
        INDArray currInput = inputs[0];
        if(layerPreProcessor != null){
            currInput = layerPreProcessor.preProcess(currInput, graph.batchSize());
        }
        return layer.activate(currInput,training);
    }

    @Override
    public Pair<Gradient,INDArray[]> doBackward(boolean tbptt, int tbpttBackwardLength){
        if(!canDoBackward()) throw new IllegalStateException("Cannot do backward pass: all epsilons not set");

        INDArray epsTotal = null;
        if(epsilons != null && epsilons.length == 1 ) epsTotal = epsilons[0];
        else if(epsilons != null && epsilons.length > 1 ){
            //TODO: check the math on this... I think it's correct though
            //This is the "output connected to multiple other layers" case
            epsTotal = epsilons[0].dup();
            for( int i=1; i<epsilons.length; i++ ){
                epsTotal.addi(epsilons[i]);
            }
        }

        Pair<Gradient,INDArray> pair;
        if(tbptt && layer instanceof BaseRecurrentLayer<?>){
            //Truncated BPTT for recurrent layers
            pair = ((BaseRecurrentLayer<?>)layer).tbpttBackpropGradient(epsTotal, tbpttBackwardLength);
        } else {
            //Normal backprop
            pair = layer.backpropGradient(epsTotal);    //epsTotal may be null for OutputLayers
        }

        if(layerPreProcessor != null){
            INDArray eps = pair.getSecond();
            eps = layerPreProcessor.backprop(eps,graph.batchSize());
            pair.setSecond(eps);
        }

        //Layers always have single activations input -> always have single epsilon output during backprop
        return new Pair<>(pair.getFirst(), new INDArray[]{pair.getSecond()});
    }


    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append("LayerVertex(id=").append(vertexIndex).append(",name=\"").append(vertexName)
                .append("\",inputs=").append(Arrays.toString(inputVertices)).append(",outputs=").append(Arrays.toString(outputVertices))
                .append(")");
        return sb.toString();
    }

}
