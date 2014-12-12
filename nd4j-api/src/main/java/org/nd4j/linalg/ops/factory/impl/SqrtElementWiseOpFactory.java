package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.Sqrt;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class SqrtElementWiseOpFactory extends BaseElementWiseOpFactory {
    private static ElementWiseOp INSTANCE = new Sqrt();
    @Override
    public ElementWiseOp create() {
        return INSTANCE;
    }
}
