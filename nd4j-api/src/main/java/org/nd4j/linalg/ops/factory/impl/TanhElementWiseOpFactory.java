package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.Tanh;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class TanhElementWiseOpFactory extends BaseElementWiseOpFactory {
    private static ElementWiseOp INSTANCE = new Tanh();
    @Override
    public ElementWiseOp create() {
        return INSTANCE;
    }
}
