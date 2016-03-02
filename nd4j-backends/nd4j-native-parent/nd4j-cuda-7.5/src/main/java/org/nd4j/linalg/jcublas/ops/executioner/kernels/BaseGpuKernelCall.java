package org.nd4j.linalg.jcublas.ops.executioner.kernels;

import jcuda.utils.KernelLauncher;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.gpumetrics.GpuMetrics;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.util.CudaArgs;
import org.nd4j.linalg.jcublas.util.PointerUtil;

/**
 * Base class for the kernel
 * calls
 *
 * @author Adam Gibson
 */
public abstract class BaseGpuKernelCall implements GpuKernelCall {
    protected CudaContext cudaContext;
    protected Object[] args;
    protected GpuMetrics metrics;
    protected Op op;
    protected static JCudaBuffer dummyDoublePointer;
    protected static JCudaBuffer dummyFloatPointer;

    static {
        dummyFloatPointer = KernelFunctions.alloc(new float[]{1});
        dummyDoublePointer = KernelFunctions.alloc(new double[]{1});
    }

    public BaseGpuKernelCall(Op op) {
        this.op = op;
        createMetrics();
    }

    @Override
    public CudaContext cudaContext() {
        return cudaContext;
    }

    @Override
    public Object[] getArgs() {
        return args;
    }

    @Override
    public Op op() {
        return op;
    }

    @Override
    public GpuMetrics metrics() {
        return metrics;
    }

    @Override
    public void invoke(String functionName) {
        /**
         * Invoke a cuda kernel by name. This will be wrt the function name.
         * Functions that are accumulations or transforms have names that end with _strided.
         *
         */

        metrics.validate();
        //module name is the op, function name is transform
        KernelFunctions.invoke(
                metrics,
                true
                , functionName
                , getType(op), cudaContext
                , args);
    }

    @Override
    public String functionName() {
        return CudaArgs.getModuleNameFor(op);
    }

    @Override
    public String moduleName() {
        return CudaArgs.getModuleNameFor(op);
    }

    @Override
    public void invoke() {
        String functionName = CudaArgs.getModuleNameFor(op);
        invoke(functionName);
    }


    private JCudaBuffer dummyDouble() {
        return dummyDoublePointer;
    }

    private JCudaBuffer dummyFloat() {
        return dummyFloatPointer;
    }


    /**
     * Converts the given parameters
     * in to extra arguments to
     * pass to the kernel
     *
     * @param extraArgs the extra arguments
     * @param dataType  the data type
     * @return
     */
    protected JCudaBuffer toArgs(Object[] extraArgs, DataBuffer.Type dataType) {
        if (dataType.equals(DataBuffer.Type.DOUBLE)) {
            if (extraArgs == null || extraArgs.length < 1)
                return dummyDouble();
            return KernelFunctions.alloc(PointerUtil.toDoubles(extraArgs));
        } else if (dataType.equals(DataBuffer.Type.FLOAT)) {
            if (extraArgs == null || extraArgs.length < 1)
                return dummyFloat();
            return KernelFunctions.alloc(PointerUtil.toFloats(extraArgs));
        }
        throw new IllegalArgumentException("Illegal datatype");
    }

    protected DataBuffer.Type getType(Op op) {
        return op.x().data().dataType(); // == DataBuffer.Type.DOUBLE ? "double" : "float";
    }


}
