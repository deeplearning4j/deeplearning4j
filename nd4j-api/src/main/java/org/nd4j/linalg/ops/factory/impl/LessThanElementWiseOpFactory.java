package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.LessThan;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class LessThanElementWiseOpFactory extends BaseElementWiseOpFactory {
    private static ElementWiseOp INSTANCE = new LessThan();
    @Override
    public ElementWiseOp create() {
        return INSTANCE;
    }
}
