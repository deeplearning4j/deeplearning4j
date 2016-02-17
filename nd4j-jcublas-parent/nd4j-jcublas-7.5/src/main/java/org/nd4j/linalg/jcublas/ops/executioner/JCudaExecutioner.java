/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.jcublas.ops.executioner;


import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDimensions;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.CopyOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.GpuKernelCall;
import org.nd4j.linalg.jcublas.ops.executioner.kernels.GpuKernelCallFactories;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;


/**
 * JCuda executioner.
 * <p/>
 * Runs ops directly on the gpu
 *
 * If requested Op doesn't exist within GPU context, DefaultOpExecutioner will be used, with arrays/buffers updated after that.
 *
 * @author Adam Gibson
 * @author raver119@gmail.com
 */
public class JCudaExecutioner extends DefaultOpExecutioner {

    private static final Allocator allocator = AtomicAllocator.getInstance();

    public JCudaExecutioner() {
    }

    @Override
    public INDArray exec(Accumulation op, int... dimension) {
        for(int i = 0; i < dimension.length; i++) {
            if(dimension[i] < 0)
                dimension[i] += op.x().rank();
        }
        //do op along all dimensions
        if(dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};


        if(op.isPassThrough()) {
            op.exec(dimension);
            return op.z();
        }


        if(dimension[0] == Integer.MAX_VALUE) {
            if(op.x() instanceof IComplexNDArray) {
                return Nd4j.scalar(execAndReturn(op).getFinalResultComplex());
            }
            return Nd4j.scalar(execAndReturn(op).getFinalResult().doubleValue());
        }

        if(op instanceof IComplexNDArray) {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if(retShape.length == 1) {
                if(dimension[0] == 0)
                    retShape = new int[] {1,retShape[0]};
                else
                    retShape = new int[] {retShape[0],1};

            }
            else if(retShape.length == 0) {
                retShape = new int[] {1,1};
            }

            IComplexNDArray ret = Nd4j.createComplex(retShape);
            IComplexNDArray linear = ret;
            for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                IComplexNumber result = execAndReturn((Accumulation) op2).getFinalResultComplex();
                linear.putScalar(i, result);

            }

            if(ret.ordering() == 'c')
                ret.setStride(ArrayUtil.reverseCopy(ret.stride()));


            return ret;
        }

        else {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if(retShape.length == 1) {
                if(dimension[0] == 0)
                    retShape = new int[] {1,retShape[0]};
                else
                    retShape = new int[] {retShape[0],1};

            }
            else if(retShape.length == 0) {
                retShape = new int[] {1,1};
            }

            //nothing to reduce
            if(ArrayUtil.prod(retShape) == op.x().length())
                return op.x();
            invoke(op,dimension);
            if(op.z() == null)
                throw new IllegalStateException("No result set");
            return op.z();
        }


    }

    @Override
    public INDArray exec(IndexAccumulation op, int... dimension) {
        for(int i = 0; i < dimension.length; i++) {
            if(dimension[i] < 0)
                dimension[i] += op.x().rank();
        }
        //do op along all dimensions
        if(dimension.length == op.x().rank())
            dimension = new int[] {Integer.MAX_VALUE};


        if(op.isPassThrough()) {
            op.exec(dimension);
            return op.z();
        }


        if(dimension[0] == Integer.MAX_VALUE) {
            if(op.x() instanceof IComplexNDArray)
                return Nd4j.scalar(execAndReturn(op).getFinalResult());
            return Nd4j.scalar(execAndReturn(op).getFinalResult());
        }

        if(op instanceof IComplexNDArray) {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if(retShape.length == 1) {
                if(dimension[0] == 0)
                    retShape = new int[] {1,retShape[0]};
                else
                    retShape = new int[] {retShape[0],1};

            }
            else if(retShape.length == 0) {
                retShape = new int[] {1,1};
            }

            IComplexNDArray ret = Nd4j.createComplex(retShape);
            IComplexNDArray linear = ret;
            for (int i = 0; i < op.x().tensorssAlongDimension(dimension); i++) {
                Op op2 = op.opForDimension(i, dimension);
                IComplexNumber result = execAndReturn((Accumulation) op2).getFinalResultComplex();
                linear.putScalar(i, result);

            }

            if(ret.ordering() == 'c')
                ret.setStride(ArrayUtil.reverseCopy(ret.stride()));


            return ret;
        }

        else {
            int[] retShape = ArrayUtil.removeIndex(op.x().shape(), dimension);
            //ensure vector is proper shape
            if(retShape.length == 1) {
                if(dimension[0] == 0)
                    retShape = new int[] {1,retShape[0]};
                else
                    retShape = new int[] {retShape[0],1};

            }
            else if(retShape.length == 0) {
                retShape = new int[] {1,1};
            }

            //nothing to reduce
            if(ArrayUtil.prod(retShape) == op.x().length())
                return op.x();

            INDArray retArray = Nd4j.create(retShape);
            invoke(op,dimension,retArray);
            return retArray;
        }


    }


    @Override
    public INDArray execAndReturn(TransformOp op, int... dimension) {
        return super.execAndReturn(op, dimension);
    }



    @Override
    public INDArray execAndReturn(ScalarOp op, int... dimension) {
        return super.execAndReturn(op, dimension);
    }

    @Override
    public Op exec(Op op, int... dimension) {
        return super.exec(op, dimension);
    }


    @Override
    public Op exec(Op op) {
        //linear views and oblong offsets can't be handled by the gpu (due to the way the buffers are interpreted as vectors)
        if(op.x() instanceof IComplexNDArray || executionMode() == ExecutionMode.JAVA || op.isPassThrough() || op instanceof CopyOp) {
            try {
                if (op.x() != null) allocator.synchronizeHostData(op.x());
                if (op.y() != null) allocator.synchronizeHostData(op.y());
                if (op.z() != null) allocator.synchronizeHostData(op.z());

                super.exec(op);
                return null;
            } finally {
                // FIXME: we need to track which op field contains result, and tickHost only for it
                if (op.x() != null) allocator.tickHostWrite(op.x());
                if (op.y() != null) allocator.tickHostWrite(op.y());
                if (op.z() != null) allocator.tickHostWrite(op.z());
            }
        }

        if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            invoke(t);
        } else if (op instanceof Accumulation) {
            Accumulation acc = (Accumulation) op;
            invoke(acc,null);
        } else if (op instanceof ScalarOp) {
            ScalarOp sc = (ScalarOp) op;
            invoke(sc);
        } else if(op instanceof BroadcastOp) {
            BroadcastOp broadcastOp = (BroadcastOp) op;
            invoke(broadcastOp);
        }
        else if(op instanceof IndexAccumulation) {
            IndexAccumulation indexAccumulation = (IndexAccumulation) op;
            invoke(indexAccumulation,null,Nd4j.scalar(0));
        }
        return op;
    }



    @Override
    public INDArray execAndReturn(TransformOp op) {
        invoke(op);
        return op.z();
    }







    private CudaContext invoke(BroadcastOp op) {
        if(!KernelFunctionLoader.getInstance().exists(op) || executionMode() == ExecutionMode.JAVA || op.isPassThrough() || op instanceof CopyOp) {
            try {
                if (op.x() != null) allocator.synchronizeHostData(op.x());
                if (op.y() != null) allocator.synchronizeHostData(op.y());
                if (op.z() != null) allocator.synchronizeHostData(op.z());

                super.exec(op);
                return null;
            } finally {
                // FIXME: we need to track which op field contains result, and tickHost only for it
                if (op.x() != null) allocator.tickHostWrite(op.x());
                if (op.y() != null) allocator.tickHostWrite(op.y());
                if (op.z() != null) allocator.tickHostWrite(op.z());
            }
        }

        try {
            CudaContext ctx;

            //total number of times to repeat each value over an element wise stride on the gpu
            int[] dimensions = op.getDimension() == null ? BroadcastDimensions.getDimensions(op.y().shape()) : op.getDimension();
            GpuKernelCall kernelCall = GpuKernelCallFactories.getFactory(op).create(op, dimensions);
            kernelCall.invoke();
            return kernelCall.cudaContext();
        } finally {
            // FIXME: we need to track which op field contains result, and tackDeviceWrite only for it
            if (op.x() != null) allocator.tackDevice(op.x());
            if (op.y() != null) allocator.tackDevice(op.y());
            if (op.z() != null) allocator.tackDevice(op.z());
        }
    }



    private CudaContext invoke(IndexAccumulation op,int[] dimension,INDArray result)  {
        if(!KernelFunctionLoader.getInstance().exists(op) || executionMode() == ExecutionMode.JAVA) {
            try {
                if (op.x() != null) allocator.synchronizeHostData(op.x());
                if (op.y() != null) allocator.synchronizeHostData(op.y());
                if (op.z() != null) allocator.synchronizeHostData(op.z());

                super.exec(op);
                return null;
            } finally {
                // FIXME: we need to track which op field contains result, and tickHost only for it
                if (op.x() != null) allocator.tickHostWrite(op.x());
                if (op.y() != null) allocator.tickHostWrite(op.y());
                if (op.z() != null) allocator.tickHostWrite(op.z());
            }
        }

        try {
            CudaContext ctx;
            GpuKernelCall accKernelCall = GpuKernelCallFactories.getFactory(op).create(op, dimension, result);
            accKernelCall.invoke();
            ctx = accKernelCall.cudaContext();
            if (result.isScalar())
                result.putScalar(0, op.getFinalResult());

            return ctx;
        } finally {
            // FIXME: we need to track which op field contains result, and tackDeviceWrite only for it
            if (op.x() != null) allocator.tackDevice(op.x());
            if (op.y() != null) allocator.tackDevice(op.y());
            if (op.z() != null) allocator.tackDevice(op.z());
        }
    }


    private CudaContext invoke(Accumulation op,int[] dimension)  {
        if(!KernelFunctionLoader.getInstance().exists(op) || executionMode() == ExecutionMode.JAVA) {
            try {
                if (op.x() != null) allocator.synchronizeHostData(op.x());
                if (op.y() != null) allocator.synchronizeHostData(op.y());
                if (op.z() != null) allocator.synchronizeHostData(op.z());

                super.exec(op);
                return null;
            } finally {
                // FIXME: we need to track which op field contains result, and tickHost only for it
                if (op.z() != null) allocator.tickHostWrite(op.z());
            }
        }

        try {
            CudaContext ctx;
            GpuKernelCall accKernelCall = GpuKernelCallFactories.getFactory(op).create(op, dimension);
            accKernelCall.invoke();
            ctx = accKernelCall.cudaContext();
            return ctx;
        } finally {
            // FIXME: we need to track which op field contains result, and tackDeviceWrite only for it
            if (op.x() != null) allocator.tackDevice(op.x());
            if (op.y() != null) allocator.tackDevice(op.y());
            if (op.z() != null) allocator.tackDevice(op.z());
        }
    }


    private CudaContext invoke(ScalarOp op) {
        if(!KernelFunctionLoader.getInstance().exists(op)  || executionMode() == ExecutionMode.JAVA) {
            try {
                if (op.x() != null) allocator.synchronizeHostData(op.x());
                if (op.y() != null) allocator.synchronizeHostData(op.y());
                if (op.z() != null) allocator.synchronizeHostData(op.z());

                super.exec(op);
                return null;
            } finally {
                // FIXME: we need to track which op field contains result, and tickHost only for it
                if (op.x() != null) allocator.tickHostWrite(op.x());
                if (op.y() != null) allocator.tickHostWrite(op.y());
                if (op.z() != null) allocator.tickHostWrite(op.z());
            }
        }

        try {
            GpuKernelCall kernelCall = GpuKernelCallFactories.getFactory(op).create(op);
            kernelCall.invoke();
            return kernelCall.cudaContext();
        } finally {
            // FIXME: we need to track which op field contains result, and tackDeviceWrite only for it
            if (op.x() != null) allocator.tackDevice(op.x());
            if (op.y() != null) allocator.tackDevice(op.y());
            if (op.z() != null) allocator.tackDevice(op.z());
        }
    }

    private CudaContext invoke(TransformOp op) {
        if(!KernelFunctionLoader.getInstance().exists(op) || op.x() instanceof IComplexNDArray || op.isPassThrough()) {

          //  return null;
            try {
                if (op.x() != null) allocator.synchronizeHostData(op.x());
                if (op.y() != null) allocator.synchronizeHostData(op.y());
                if (op.z() != null) allocator.synchronizeHostData(op.z());

                super.exec(op);
                return null;
            } finally {
                // FIXME: we need to track which op field contains result, and tickHost only for it
                if (op.x() != null) allocator.tickHostWrite(op.x());
                if (op.y() != null) allocator.tickHostWrite(op.y());
                if (op.z() != null) allocator.tickHostWrite(op.z());
            }
        }

        try {
            GpuKernelCall kernelCall = GpuKernelCallFactories.getFactory(op).create(op);
            kernelCall.invoke();
            return kernelCall.cudaContext();
        } finally {
            // FIXME: we need to track which op field contains result, and tackDeviceWrite only for it
            if (op.x() != null) allocator.tackDevice(op.x());
            if (op.y() != null) allocator.tackDevice(op.y());
            if (op.z() != null) allocator.tackDevice(op.z());
        }
    }
}


