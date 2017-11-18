package org.nd4j.linalg.api.ops.impl.shape;

/**
 * Stack op conversion
 *
 * @author raver119@gmail.com
 */
public class Pack extends Stack {
    @Override
    public String opName() {
        return "pack";
    }
}
