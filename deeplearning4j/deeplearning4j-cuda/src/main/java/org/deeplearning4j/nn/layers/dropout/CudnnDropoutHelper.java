package org.deeplearning4j.nn.layers.dropout;

import org.bytedeco.javacpp.*;
import org.deeplearning4j.nn.conf.dropout.DropoutHelper;
import org.deeplearning4j.nn.layers.BaseCudnnHelper;
import org.nd4j.jita.allocator.Allocator;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.util.ArrayUtil;

import static org.bytedeco.javacpp.cudnn.*;
import static org.bytedeco.javacpp.cudnn.cudnnDestroyTensorDescriptor;

public class CudnnDropoutHelper extends BaseCudnnHelper implements DropoutHelper {

    private static class CudnnDropoutContext extends CudnnContext {

        private static class Deallocator extends CudnnDropoutContext implements Pointer.Deallocator {
            Deallocator(CudnnDropoutContext c) {
                super(c);
            }

            @Override
            public void deallocate() {
                destroyHandles();
            }
        }

        private cudnn.cudnnTensorStruct xTensorDesc = new cudnn.cudnnTensorStruct();    //Input
        private cudnn.cudnnTensorStruct dxTensorDesc = new cudnn.cudnnTensorStruct();   //Grad at input
        private cudnn.cudnnTensorStruct yTensorDesc = new cudnn.cudnnTensorStruct();    //Output
        private cudnn.cudnnTensorStruct dyTensorDesc = new cudnn.cudnnTensorStruct();   //Grad at output
        private cudnn.cudnnDropoutStruct dropoutDesc = new cudnn.cudnnDropoutStruct();

        public CudnnDropoutContext() {
            createHandles();
            deallocator(new Deallocator(this));
        }

        public CudnnDropoutContext(CudnnDropoutContext c) {
            super(c);
            xTensorDesc = new cudnn.cudnnTensorStruct(c.xTensorDesc);
            dxTensorDesc = new cudnn.cudnnTensorStruct(c.dxTensorDesc);
            yTensorDesc = new cudnn.cudnnTensorStruct(c.yTensorDesc);
            dyTensorDesc = new cudnn.cudnnTensorStruct(c.dyTensorDesc);
            dropoutDesc = new cudnn.cudnnDropoutStruct(c.dropoutDesc);
        }

        @Override
        protected void createHandles() {
            super.createHandles();
            checkCudnn(cudnnCreateTensorDescriptor(xTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(dxTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(yTensorDesc));
            checkCudnn(cudnnCreateTensorDescriptor(dyTensorDesc));
            checkCudnn(cudnnCreateDropoutDescriptor(dropoutDesc));
        }

        @Override
        protected void destroyHandles() {
            checkCudnn(cudnnDestroyTensorDescriptor(xTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(dxTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(yTensorDesc));
            checkCudnn(cudnnDestroyTensorDescriptor(dyTensorDesc));
            checkCudnn(cudnnDestroyDropoutDescriptor(dropoutDesc));
            super.destroyHandles();
        }
    }

    private CudnnDropoutContext cudnnContext = new CudnnDropoutContext();
    private Pointer statesPtr;
    private long stateSizeBytes;
    private long seed;


    @Override
    public void applyDropout(INDArray input, INDArray resultArray, double dropoutInputRetainProb) {
        float p = (float)(1.0 - dropoutInputRetainProb);    //CuDNN uses p = probability of setting to 0

        checkCudnn(cudnnSetTensorNdDescriptor(cudnnContext.xTensorDesc, dataType, input.rank(),
                //TODO int cast
                ArrayUtil.toInts(input.shape()),
                ArrayUtil.toInts(input.stride())));
        checkCudnn(cudnnSetTensorNdDescriptor(cudnnContext.yTensorDesc, dataType, resultArray.rank(),
                //TODO int cast
                ArrayUtil.toInts(resultArray.shape()),
                ArrayUtil.toInts(resultArray.stride())));

        //Reserve space
        SizeTPointer stateSizeBytesPtr = new SizeTPointer();        //TODO correct???
        stateSizeBytes = stateSizeBytesPtr.get();
        checkCudnn(cudnnDropoutGetReserveSpaceSize(cudnnContext.xTensorDesc, stateSizeBytesPtr));

        //Dropout descriptor:
        if(statesPtr == null || statesPtr.capacity() != stateSizeBytes){
            //TODO: Is this correct? "Pointer to user-allocated GPU memory that will hold random number generator states."
            statesPtr = new BytePointer(stateSizeBytes);
        }
        long seed = Nd4j.getRandom().nextLong();
        checkCudnn(cudnnSetDropoutDescriptor(cudnnContext.dropoutDesc, cudnnContext, p, statesPtr, stateSizeBytes, seed));


        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(input, resultArray);
        Pointer xPtr = allocator.getPointer(input, context);
        Pointer yPtr = allocator.getPointer(resultArray, context);

        checkCudnn(cudnnSetStream(cudnnContext, new cuda.CUstream_st(context.getOldStream())));
        checkCudnn(cudnnDropoutForward(cudnnContext, cudnnContext.dropoutDesc, cudnnContext.xTensorDesc, xPtr,
                cudnnContext.yTensorDesc, yPtr, statesPtr, stateSizeBytes));

        allocator.registerAction(context, input, resultArray);
        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            context.syncOldStream();
    }

    @Override
    public void backprop(INDArray gradAtOutput, INDArray gradAtInput) {
        //dropout descriptor should already be set; don't need to set it again?
        //checkCudnn(cudnnSetDropoutDescriptor(cudnnContext.dropoutDesc, cudnnContext, p, statesPtr, stateSizeBytes, seed));


        checkCudnn(cudnnSetTensorNdDescriptor(cudnnContext.dyTensorDesc, dataType, gradAtOutput.rank(),
                //TODO int cast
                ArrayUtil.toInts(gradAtOutput.shape()),
                ArrayUtil.toInts(gradAtOutput.stride())));
        checkCudnn(cudnnSetTensorNdDescriptor(cudnnContext.dxTensorDesc, dataType, gradAtInput.rank(),
                //TODO int cast
                ArrayUtil.toInts(gradAtInput.shape()),
                ArrayUtil.toInts(gradAtInput.stride())));

        Allocator allocator = AtomicAllocator.getInstance();
        CudaContext context = allocator.getFlowController().prepareAction(gradAtOutput, gradAtInput);
        Pointer dyPtr = allocator.getPointer(gradAtOutput, context);
        Pointer dxPtr = allocator.getPointer(gradAtInput, context);

        checkCudnn(cudnnDropoutBackward(cudnnContext, cudnnContext.dropoutDesc, cudnnContext.dyTensorDesc, dyPtr,
                cudnnContext.dxTensorDesc, dxPtr, statesPtr, stateSizeBytes));

        allocator.registerAction(context, gradAtOutput, gradAtInput);
        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            context.syncOldStream();
    }
}
