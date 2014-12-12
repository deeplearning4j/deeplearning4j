package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.Max;
import org.nd4j.linalg.ops.transforms.Min;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class MinElementWiseOpFactory extends BaseElementWiseOpFactory {
    private double max = 0.0;
    private static Map<Double,ElementWiseOp> MIN_FUNCTIONS = new ConcurrentHashMap<>();


    @Override
    public ElementWiseOp create(Object[] args) {
        if(args != null && args.length > 0)
            this.max = (double) args[0];
        if(MIN_FUNCTIONS.containsKey(max))
            return MIN_FUNCTIONS.get(max);
        else {
            ElementWiseOp ret = new Min(max);
            MIN_FUNCTIONS.put(max,ret);
            return ret;
        }
    }

    @Override
    public ElementWiseOp create() {
        if(MIN_FUNCTIONS.containsKey(max))
            return MIN_FUNCTIONS.get(max);
        else {
            ElementWiseOp ret = new Min(max);
            MIN_FUNCTIONS.put(max,ret);
            return ret;
        }
    }
}
