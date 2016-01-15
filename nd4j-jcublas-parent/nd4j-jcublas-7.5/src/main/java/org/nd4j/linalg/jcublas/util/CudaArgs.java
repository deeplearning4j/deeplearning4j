package org.nd4j.linalg.jcublas.util;


import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import jcuda.Pointer;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Handles conversion of
 * arguments passed to jcuda
 * to their proper primitives
 * when invoked with pointers.
 *
 * @author Adam Gibson
 */
public class CudaArgs {
    private CudaArgs() {}

    /**
     * For invoking a cuda kernel
     * this returns the module name for the given op
     * @param op the op to get the module name for
     * @return the module name for the given op
     */
    public static String getModuleNameFor( Op op) {
        //String functionName = op instanceof TransformOp || op instanceof Accumulation || op instanceof IndexAccumulation ? op.name() + "_strided" : op.name();
        String moduleName = null;
        if (op instanceof Accumulation) {
            moduleName = "reduce";
        } else if (op instanceof TransformOp) {
            // FIXME: we need special case for pairwise transforms for now. Later we should make them separate kernel call
            if (op.name().equals("add")) {
                moduleName = "pairWiseTransform";
            } else if (op.name().equals("copy")) {
                moduleName = "pairWiseTransform";
            } else if (op.name().equals("div")) {
                moduleName = "pairWiseTransform";
            } else if (op.name().equals("mul")) {
                moduleName = "pairWiseTransform";
            } else if (op.name().equals("rdiv")) {
                moduleName = "pairWiseTransform";
            } else if (op.name().equals("rsub")) {
                moduleName = "pairWiseTransform";
            } else if (op.name().equals("sub")) {
                moduleName = "pairWiseTransform";

            } else {
                moduleName = "transform";
            }
        } else if (op instanceof ScalarOp) {
            moduleName = "scalar";
        } else if (op instanceof  BroadcastOp) {
            moduleName = "broadcast";
        }
        return moduleName;
    }

    public static int getOpCode(Op op) {
        int code = -1;

        String name = op.name();

        if (op instanceof Accumulation) {
            if (name.equals("mean")) {
                code = 0;
            } else if (name.equals("sum")) {
                code = 1;
            } else if (name.equals("bias")) {
                code = 2;
            } else if (name.equals("max")) {
                code = 3;
            } else if (name.equals("min")) {
                code = 4;
            } else if (name.equals("norm1")) {
                code = 5;
            } else if (name.equals("norm2")) {
                code = 6;
            } else if (name.equals("normmax")) {
                code = 7;
            } else if (name.equals("prod")) {
                code = 8;
            } else if (name.equals("std")) {
                code = 9;
            } else if (name.equals("var")) {
                code = 10;
            }
        } else if (op instanceof TransformOp) {
            if (name.equals("abs")) {
                code = 0;
            } else if (name.equals("ceil")) {
                code = 1;
            } else if (name.equals("cos")) {
                code = 2;
            } else if (name.equals("exp")) {
                code = 3;
            } else if (name.equals("floor")) {
                code = 4;
            } else if (name.equals("log")) {
                code = 5;
            } else if (name.equals("neg")) {
                code = 6;
            } else if (name.equals("pow")) {
                code = 7;
            } else if (name.equals("round")) {
                code = 8;
            } else if (name.equals("setrange")) {
                code = 9;
            } else if (name.equals("sigmoid")) {
                code = 10;
            } else if (name.equals("sign")) {
                code = 11;
            } else if (name.equals("sin")) {
                code = 12;
            } else if (name.equals("softplus")) {
                code = 13;
            } else if (name.equals("sqrt")) {
                code = 14;
            } else if (name.equals("tanh")) {
                code = 15;
            } else if (name.equals("acos")) {
                code = 16;
            } else if (name.equals("asin")) {
                code = 17;
            } else if (name.equals("atan")) {
                code = 18;

                // FIXME: we need special case for pairwise transforms for now. Later we should make them separate kernel call
            } else if (name.equals("add")) {
                code = 0;
            } else if (name.equals("copy")) {
                code = 1;
            } else if (name.equals("div")) {
                code = 2;
            } else if (name.equals("div")) {
                code = 3;
            } else if (name.equals("eq")) {
                code = 4;
            } else if (name.equals("gt")) {
                code = 5;
            } else if (name.equals("lt")) {
                code = 6;
            } else if (name.equals("mul")) {
                code = 7;
            } else if (name.equals("div")) {
                code = 8;
            } else if (name.equals("driv")) {
                code = 9;
            } else if (name.equals("rsub")) {
                code = 10;
            } else if (name.equals("sub")) {
                code = 11;
            }
        } else if (op instanceof ScalarOp) {
            if (name.startsWith("add")) {
                code = 0;
            } else if (name.startsWith("sub")) {
                code =  1;
            } else if (name.startsWith("mul")) {
                code =  2;
            } else if (name.startsWith("div")) {
                code =  3;
            } else if (name.startsWith("rdiv")) {
                code =  4;
            } else if (name.startsWith("rsub")) {
                code =  5;
            } else if (name.startsWith("max")) {
                code =  6;
            } else if (name.startsWith("lessthan")) {
                code =  7;
            } else if (name.startsWith("greaterthan")) {
                code =  8;
            } else if (name.startsWith("eq")) {
                code =  9;
            } else if (name.startsWith("lte")) {
                code =  10;
            } else if (name.startsWith("neq")) {
                code =  11;
            } else if (name.startsWith("min")) {
                code =  12;
            } else if (name.startsWith("set")) {
                code =  13;
            }
        } else if (op instanceof  BroadcastOp) {
            if (name.equals("broadcastadd")) {
                code = 0;
            } else if (name.equals("broadcastsub")) {
                code = 1;
            } else if (name.equals("broadcastmul")) {
                code = 2;
            } else if (name.equals("broadcastdiv")) {
                code = 3;
            } else if (name.equals("broadcastrdiv")) {
                code = 4;
            } else if (name.equals("broadcastrsub")) {
                code = 5;
            } else if (name.equals("broadcastcopy")) {
                code = 6;
            }
        }

        return code;
    }




    /**
     *
     * @param context
     * @param kernelParams
     * @return
     */
    public static ArgsAndReferences argsAndReference(CudaContext context,Object...kernelParams) {
        Map<Object, Object> idMap = new IdentityHashMap<>();
        Object[] kernelParameters = new Object[kernelParams.length];
        List<CublasPointer> pointersToFree = new ArrayList<>();
        Multimap<INDArray, CublasPointer> arrayToPointer = ArrayListMultimap.create();
        for (int i = 0; i < kernelParams.length; i++) {
            Object arg = kernelParams[i];

            // If the instance is a JCudaBuffer we should assign it to the device
            if (arg instanceof JCudaBuffer) {
                JCudaBuffer buffer = (JCudaBuffer) arg;
                if (!idMap.containsKey(buffer)) {
                    CublasPointer pointerToFree = new CublasPointer(buffer, context);
                    kernelParameters[i] = pointerToFree.getDevicePointer();
                    pointersToFree.add(pointerToFree);
                    idMap.put(buffer, pointerToFree.getDevicePointer());
                } else {
                    Pointer pointer = (Pointer) idMap.get(buffer);
                    kernelParameters[i] = pointer;
                }

            } else if (arg instanceof INDArray) {
                INDArray array = (INDArray) arg;
                if (!idMap.containsKey(array)) {
                    CublasPointer pointerToFree = new CublasPointer(array, context);
                    kernelParameters[i] = pointerToFree.getDevicePointer();
                    pointersToFree.add(pointerToFree);
                    arrayToPointer.put(array, pointerToFree);
                    idMap.put(array, pointerToFree.getDevicePointer());
                } else {
                    Pointer pointer = (Pointer) idMap.get(array);
                    kernelParameters[i] = pointer;
                }

            } else {
                kernelParameters[i] = arg;
            }

        }

        return new ArgsAndReferences(kernelParameters,idMap,pointersToFree,arrayToPointer);
    }


    @Data
    @AllArgsConstructor
    public static class ArgsAndReferences {
        private Object[] args;
        private Map<Object,Object> idMap;
        private List<CublasPointer> pointersToFree;
        /**
         * conversion list of arrays to their assigned cublas pointer
         */
        private Multimap<INDArray, CublasPointer> arrayToPointer;


    }

}
