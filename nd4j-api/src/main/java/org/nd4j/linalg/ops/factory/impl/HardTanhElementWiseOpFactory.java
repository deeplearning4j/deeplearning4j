package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.HardTanh;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class HardTanhElementWiseOpFactory extends BaseElementWiseOpFactory {

    private static ElementWiseOp INSTANCE = new HardTanh();

    @Override
    public ElementWiseOp create() {
        return INSTANCE;
    }
}
