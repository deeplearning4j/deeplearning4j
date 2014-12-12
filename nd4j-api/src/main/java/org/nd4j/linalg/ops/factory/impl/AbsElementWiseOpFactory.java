package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.Abs;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class AbsElementWiseOpFactory extends BaseElementWiseOpFactory {
    private static ElementWiseOp INSTANCE = new Abs();

    @Override
    public ElementWiseOp create() {
        return INSTANCE;
    }

    @Override
    public ElementWiseOp create(Object[] args) {
        return create();
    }
}
