package org.deeplearning4j.nn.graph.vertex.impl;

import java.io.Serializable;
import java.util.regex.Pattern;

/**
 * VertexKey uniquely defines a GraphVertex in a ComputationGraph.
 * There could be multiple vertices sharing the same layer.
 */

public class VertexKey implements Serializable {
    String layerName;
    int streamIndex;

    public VertexKey(String layerName, int streamIndex){
        this.layerName = layerName;
        this.streamIndex = streamIndex;
    }

    public String toString(){
        if (streamIndex == 0)
            return layerName;
        return layerName + ":" + streamIndex;
    }

    public static VertexKey fromString(String s){
        String[] sp = s.split(Pattern.quote(":"));
        return create(sp[0], Integer.parseInt(sp[1]));
    }

    public String getLayerName(){
        return layerName;
    }

    public int getStreamIndex(){
        return streamIndex;
    }

    public static VertexKey create(String layerName, int streamIndex){
        return new VertexKey(layerName, streamIndex);
    }

    public static VertexKey create(String layerName){
        return new VertexKey(layerName, 0);
    }
}
