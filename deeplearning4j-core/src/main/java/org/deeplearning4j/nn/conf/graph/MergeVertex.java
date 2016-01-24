package org.deeplearning4j.nn.conf.graph;


import org.deeplearning4j.nn.graph.ComputationGraph;

/** A MergeVertex is used to combine the activations of two or more layers/GraphVertex by means of concatenation/merging.<br>
 * Exactly how this is done depends on the type of input.<br>
 * For 2d (feed forward layer) inputs: MergeVertex([numExamples,layerSize1],[numExamples,layerSize2]) -> [numExamples,layerSize1 + layerSize2]<br>
 * For 3d (time series) inputs: MergeVertex([numExamples,layerSize1,timeSeriesLength],[numExamples,layerSize2,timeSeriesLength])
 *      -> [numExamples,layerSize1 + layerSize2,timeSeriesLength]<br>
 * For 4d (convolutional) inputs: MergeVertex([numExamples,depth1,width,height],[numExamples,depth2,width,height])
 *      -> [numExamples,depth1 + depth2,width,height]<br>
 * @author Alex Black
 */
public class MergeVertex extends GraphVertex {

    @Override
    public MergeVertex clone() {
        return new MergeVertex();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof MergeVertex;
    }

    @Override
    public int hashCode(){
        return 433682566;
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx) {
        return new org.deeplearning4j.nn.graph.vertex.impl.MergeVertex(graph,name,idx);
    }
}
