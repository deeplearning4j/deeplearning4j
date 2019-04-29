package org.nd4j.imports.graphmapper;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.List;
import java.util.Map;

/**
 * An interface for overriding the import of an operation
 * @author Alex Black
 */
public interface OpImportOverride<GRAPH_TYPE, NODE_TYPE, ATTR_TYPE> {

    /**
     * Initialize the operation and return its output variables
     */
    List<SDVariable> initFromTensorFlow(List<SDVariable> inputs, List<SDVariable> controlDepInputs, NODE_TYPE nodeDef, SameDiff initWith, Map<String,ATTR_TYPE> attributesForNode, GRAPH_TYPE graph);

}
