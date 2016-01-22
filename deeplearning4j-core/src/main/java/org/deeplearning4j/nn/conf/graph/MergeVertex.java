package org.deeplearning4j.nn.conf.graph;


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
}
