package org.nd4j.linalg.jcublas.ops.executioner.kernels;

import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.factory.impl.*;

/**
 * Name space class
 * for the different gpu kernel call factories.
 *
 * @author Adam Gibson
 */
public class GpuKernelCallFactories {
    private GpuKernelCallFactories() {}


    public static GpuKernelCallFactory getFactory(Op op) {
        if(op instanceof TransformOp) {
            return new TransformKernelCallFactory();
        }
        else if(op instanceof ScalarOp) {
            return new ScalarKernelCallFactory();
        }
        else if(op instanceof Accumulation) {
            return new AccumulationKernelCallFactory();
        }
        else if (op instanceof IndexAccumulation) {
            return new IndexAccumulationKernelCallFactory();
        }
        else if(op instanceof BroadcastOp) {
            return new BroadcastKernelCallFactory();
        }

        throw new IllegalStateException("Illegal type");
    }

}
