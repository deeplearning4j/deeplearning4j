package org.nd4j.linalg.jcublas.ops.executioner.kernels.args;

import org.nd4j.linalg.api.ops.Op;

/**
 * Created by agibsonccc on 12/21/15.
 */
public class KernelCallPointerUtil {
    /**
     * Returns the arguments
     * wrt current initialized pointer
     * arguments and the given operation
     *
     * This is for (based on the op type)
     * retrieving and initializing
     * device pointer references from an existing
     * array.
     *
     * @param op the operation
     *
     * @param kernelArgs the initialized kernel arguments.
     * @return
     */
    public static KernelCallPointerArgs getInitializedPointer(Op op,Object[] kernelArgs) {
        return null;
    }


}
