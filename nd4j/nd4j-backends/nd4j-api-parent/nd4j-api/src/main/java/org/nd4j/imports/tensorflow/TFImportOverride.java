package org.nd4j.imports.tensorflow;

import org.nd4j.imports.graphmapper.OpImportOverride;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

public interface TFImportOverride extends OpImportOverride<GraphDef, NodeDef, AttrValue> {

}
