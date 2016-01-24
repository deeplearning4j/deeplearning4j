package org.deeplearning4j.nn.conf.graph;


import org.deeplearning4j.nn.graph.ComputationGraph;

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
