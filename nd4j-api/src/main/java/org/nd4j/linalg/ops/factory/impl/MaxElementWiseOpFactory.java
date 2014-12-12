package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.Max;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class MaxElementWiseOpFactory extends BaseElementWiseOpFactory {

    private double max = 0.0;
    private static Map<Double,ElementWiseOp> MAX_FUNCTIONS = new ConcurrentHashMap<>();


    @Override
    public ElementWiseOp create(Object[] args) {
        if(args != null && args.length > 0)
            this.max = Double.valueOf(args[0].toString());
        if(MAX_FUNCTIONS.containsKey(max))
            return MAX_FUNCTIONS.get(max);
        else {
            ElementWiseOp ret = new Max(max);
            MAX_FUNCTIONS.put(max,ret);
            return ret;
        }
    }

    @Override
    public ElementWiseOp create() {
        if(MAX_FUNCTIONS.containsKey(max))
            return MAX_FUNCTIONS.get(max);
        else {
            ElementWiseOp ret = new Max(max);
            MAX_FUNCTIONS.put(max,ret);
            return ret;
        }
    }
}
