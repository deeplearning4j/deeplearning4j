package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.Identity;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class IdentityElementWiseOpFactory extends BaseElementWiseOpFactory {
    private static ElementWiseOp INSTANCE = new Identity();
    @Override
    public ElementWiseOp create() {
        return INSTANCE;
    }
}
