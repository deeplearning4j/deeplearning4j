package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.Sigmoid;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class SigmoidElementWiseOpFactory extends BaseElementWiseOpFactory {
   private static ElementWiseOp INSTANCE = new Sigmoid();

    @Override
    public ElementWiseOp create() {
        return INSTANCE;
    }
}
