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
            System.out.println("Transform Op");
            return new TransformKernelCallFactory();
        }
        else if(op instanceof ScalarOp) {
            System.out.println("Scalar Op");
            return new ScalarKernelCallFactory();
        }
        else if(op instanceof Accumulation) {
            System.out.println("Accumulation Op");
            return new AccumulationKernelCallFactory();
        }
        else if (op instanceof IndexAccumulation) {
            System.out.println("IndexAccumulation Op");
            return new IndexAccumulationKernelCallFactory();
        }
        else if(op instanceof BroadcastOp) {
            System.out.println("Broadcast Op");
            return new BroadcastKernelCallFactory();
        }

        throw new IllegalStateException("Illegal type");
    }

}
