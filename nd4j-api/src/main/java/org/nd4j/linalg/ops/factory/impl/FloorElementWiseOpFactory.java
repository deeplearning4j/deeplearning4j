package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.Floor;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class FloorElementWiseOpFactory extends BaseElementWiseOpFactory {
    private static ElementWiseOp INSTANCE = new Floor();
    @Override
    public ElementWiseOp create() {
        return INSTANCE;
    }
}
