package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.EqualTo;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class EqualToElementWiseOpFactory extends BaseElementWiseOpFactory {

    private static ElementWiseOp INSTANCE = new EqualTo();

    @Override
    public ElementWiseOp create() {
        return INSTANCE;
    }
}
