package org.nd4j.imports.graphmapper;

import lombok.Data;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Map;

@Data
public class ImportState<GRAPH_TYPE,TENSOR_TYPE> {
    private SameDiff sameDiff;
    private GRAPH_TYPE graph;
    private Map<String,TENSOR_TYPE> variables;


}
