package org.nd4j.imports.converters.tf;

import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.tensorflow.framework.NodeDef;

/**
 * Stack op conversion
 *
 * @author raver119@gmail.com
 */
public class Pack extends Stack {
    @Override
    public String opName() {
        return "pack";
    }
}
