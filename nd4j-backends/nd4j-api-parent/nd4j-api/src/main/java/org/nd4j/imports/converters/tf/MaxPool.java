package org.nd4j.imports.converters.tf;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TNode;
import org.tensorflow.framework.NodeDef;

/**
 * MaxPool2D op converter
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class MaxPool extends AvgPool {

    @Override
    public String opName() {
        return "maxpool";
    }
}
