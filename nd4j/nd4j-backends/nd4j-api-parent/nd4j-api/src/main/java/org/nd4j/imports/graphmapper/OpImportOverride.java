package org.nd4j.imports.graphmapper;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.List;
import java.util.Map;

public interface OpImportOverride<GRAPH_TYPE, NODE_TYPE, ATTR_TYPE> {

    List<SDVariable> initFromTensorFlow(List<SDVariable> inputs, List<SDVariable> controlDepInputs, NODE_TYPE nodeDef, SameDiff initWith, Map<String,ATTR_TYPE> attributesForNode, GRAPH_TYPE graph);

    //DataType dataTypeForOutput(int outputNum, List<SDVariable> inputs, List<SDVariable> controlDepInputs, NODE_TYPE nodeDef, SameDiff initWith, Map<String,ATTR_TYPE> attributesForNode, GRAPH_TYPE graph);

}
