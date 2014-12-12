package org.nd4j.linalg.ops.factory;

import org.nd4j.linalg.ops.ElementWiseOp;

/**
 * Create element wise operations
 *
 * @author Adam Gibson
 */
public interface ElementWiseOpFactory  {

    /**
     * Create element wise operations
     * @return the element wise operation to create
     */
    ElementWiseOp create();


    /**
     * Create with the given arguments
     * @param args
     * @return
     */
    ElementWiseOp create(Object[] args);


}
