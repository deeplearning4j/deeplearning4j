package org.nd4j.imports.converters;

import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;

/**
 * This class provides wrappers for various Graph engines. I.e. TensorFlow or PyTorch
 *
 * @author raver119@gmail.com
 */
public class ImportHelper implements NodeMapper {

    @Override
    public TNode asIntermediate(Object node, TGraph graph) {
        return null;
    }
}
