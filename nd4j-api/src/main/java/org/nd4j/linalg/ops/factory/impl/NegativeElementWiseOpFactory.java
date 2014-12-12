package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactory;
import org.nd4j.linalg.ops.transforms.Negative;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class NegativeElementWiseOpFactory extends BaseElementWiseOpFactory {
    private static ElementWiseOp INSTANCE = new Negative();

    @Override
    public ElementWiseOp create() {
        return INSTANCE;
    }
}
