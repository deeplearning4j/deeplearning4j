package org.nd4j.linalg.ops.factory.impl;

import org.nd4j.linalg.ops.ElementWiseOp;
import org.nd4j.linalg.ops.transforms.Pow;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by agibsonccc on 12/11/14.
 */
public class PowElementWiseOpFactory extends BaseElementWiseOpFactory {

    private double pow = 0.0;
    private static Map<Double,ElementWiseOp> FUNCTIONS = new ConcurrentHashMap<>();


    @Override
    public ElementWiseOp create(Object[] args) {
        if(args != null && args.length > 0)
            this.pow = Double.valueOf(args[0].toString());
        if(FUNCTIONS.containsKey(pow))
            return FUNCTIONS.get(pow);
        else {
            ElementWiseOp ret = new Pow(pow);
            FUNCTIONS.put(pow,ret);
            return ret;
        }
    }

    @Override
    public ElementWiseOp create() {
        if(FUNCTIONS.containsKey(pow))
            return FUNCTIONS.get(pow);
        else {
            ElementWiseOp ret = new Pow(pow);
            FUNCTIONS.put(pow,ret);
            return ret;
        }
    }
}
