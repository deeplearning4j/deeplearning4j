package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.Sign;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class SignElementWiseOpFactory extends BaseElementWiseOpFactory {
    private static ElementWiseOp INSTANCE = new Sign();
    @Override
    public ElementWiseOp create() {
        return INSTANCE;
    }
}
