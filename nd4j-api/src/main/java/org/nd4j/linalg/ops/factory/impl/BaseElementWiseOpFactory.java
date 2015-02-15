package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.factory.ElementWiseOpFactory;

/**
 * Default element wise operations
 *
 * @author Adam Gibson
 */
public abstract class BaseElementWiseOpFactory implements ElementWiseOpFactory {


    @Override
    public ElementWiseOp create(Object[] args) {
        return create();
    }
}
