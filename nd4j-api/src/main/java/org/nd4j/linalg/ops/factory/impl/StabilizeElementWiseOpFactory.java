package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.Stabilize;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class StabilizeElementWiseOpFactory extends BaseElementWiseOpFactory {
    private double stabilize = 0.0;
    private static Map<Double,ElementWiseOp> FUNCTIONS = new ConcurrentHashMap<>();


    @Override
    public ElementWiseOp create(Object[] args) {
        if(args != null && args.length > 0)
            this.stabilize = Double.valueOf(args[0].toString());
        if(FUNCTIONS.containsKey(stabilize))
            return FUNCTIONS.get(stabilize);
        else {
            ElementWiseOp ret = new Stabilize(stabilize);
            FUNCTIONS.put(stabilize,ret);
            return ret;
        }
    }

    @Override
    public ElementWiseOp create() {
        if(FUNCTIONS.containsKey(stabilize))
            return FUNCTIONS.get(stabilize);
        else {
            ElementWiseOp ret = new Stabilize(stabilize);
            FUNCTIONS.put(stabilize,ret);
            return ret;
        }
    }
}
