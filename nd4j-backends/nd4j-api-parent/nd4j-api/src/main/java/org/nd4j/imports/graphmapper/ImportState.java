package org.nd4j.imports.graphmapper;

import lombok.Data;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.primitives.Pair;

import java.util.Map;

@Data
public class ImportState<GRAPH_TYPE,TENSOR_TYPE> {
    private int nodeCount;
    private SameDiff sameDiff;
    private GRAPH_TYPE graph;
    private Map<String,TENSOR_TYPE> variables;
    private Map<String,Pair<int[],int[]>> vertexIdMap;

    public void incrementNodeCount() {
        nodeCount++;
    }

}
