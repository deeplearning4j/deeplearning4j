package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.GreaterThan;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class GreaterThanElementWiseOpFactory extends BaseElementWiseOpFactory {
    private static ElementWiseOp INSTANCE = new GreaterThan();
    @Override
    public ElementWiseOp create() {
        return INSTANCE;
    }
}
