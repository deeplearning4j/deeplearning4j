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
