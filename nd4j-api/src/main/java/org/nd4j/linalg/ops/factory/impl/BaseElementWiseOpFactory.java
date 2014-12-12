package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactory;

/**
 * Created by agibsonccc on 12/11/14.
 */
public abstract class BaseElementWiseOpFactory implements ElementWiseOpFactory {


    @Override
    public ElementWiseOp create(Object[] args) {
        return create();
    }
}
