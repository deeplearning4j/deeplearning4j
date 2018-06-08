/*-
 *
 *  * Copyright 2017 Skymind,Inc.
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
 */
package org.deeplearning4j.nn.layers;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.*;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.factory.Nd4j;

import static org.bytedeco.javacpp.cuda.*;
import static org.bytedeco.javacpp.cudnn.*;

/**
 * Functionality shared by all cuDNN-based helpers.
 *
 * @author saudet
 */
@Slf4j
public abstract class BaseCudnnHelper {

    protected static void checkCuda(int error) {
        if (error != cudaSuccess) {
            throw new RuntimeException("CUDA error = " + error + ": " + cudaGetErrorString(error).getString());
        }
    }

    protected static void checkCudnn(int status) {
        if (status != CUDNN_STATUS_SUCCESS) {
            throw new RuntimeException("cuDNN status = " + status + ": " + cudnnGetErrorString(status).getString());
        }
    }

    protected static class CudnnContext extends cudnnContext {

        protected static class Deallocator extends CudnnContext implements Pointer.Deallocator {
            Deallocator(CudnnContext c) {
                super(c);
            }

            @Override
            public void deallocate() {
                destroyHandles();
            }
        }

        public CudnnContext() {
            // insure that cuDNN initializes on the same device as ND4J for this thread
            Nd4j.create(1);
            AtomicAllocator.getInstance();
            // This needs to be called in subclasses:
            // createHandles();
            // deallocator(new Deallocator(this));
        }

        public CudnnContext(CudnnContext c) {
            super(c);
        }

        protected void createHandles() {
            checkCudnn(cudnnCreate(this));
        }

        protected void destroyHandles() {
            checkCudnn(cudnnDestroy(this));
        }
    }

    protected static class DataCache extends Pointer {

        static class Deallocator extends DataCache implements Pointer.Deallocator {
            Deallocator(DataCache c) {
                super(c);
            }

            @Override
            public void deallocate() {
                checkCuda(cudaFree(this));
                setNull();
            }
        }

        static class HostDeallocator extends DataCache implements Pointer.Deallocator {
            HostDeallocator(DataCache c) {
                super(c);
            }

            @Override
            public void deallocate() {
                checkCuda(cudaFreeHost(this));
                setNull();
            }
        }

        public DataCache() {}

        public DataCache(long size) {
            position = 0;
            limit = capacity = size;
            int error = cudaMalloc(this, size);
            if (error != cudaSuccess) {
                log.warn("Cannot allocate " + size + " bytes of device memory (CUDA error = " + error
                                + "), proceeding with host memory");
                checkCuda(cudaMallocHost(this, size));
                deallocator(new HostDeallocator(this));
            } else {
                deallocator(new Deallocator(this));
            }
        }

        public DataCache(DataCache c) {
            super(c);
        }
    }

    protected static class TensorArray extends PointerPointer<cudnnTensorStruct> {

        static class Deallocator extends TensorArray implements Pointer.Deallocator {
            Pointer owner;

            Deallocator(TensorArray a, Pointer owner) {
                this.address = a.address;
                this.capacity = a.capacity;
                this.owner = owner;
            }

            @Override
            public void deallocate() {
                for (int i = 0; !isNull() && i < capacity; i++) {
                    cudnnTensorStruct t = this.get(cudnnTensorStruct.class, i);
                    checkCudnn(cudnnDestroyTensorDescriptor(t));
                }
                if (owner != null) {
                    owner.deallocate();
                    owner = null;
                }
                setNull();
            }
        }

        public TensorArray() {}

        public TensorArray(long size) {
            PointerPointer p = new PointerPointer(size);
            p.deallocate(false);
            this.address = p.address();
            this.limit = p.limit();
            this.capacity = p.capacity();

            cudnnTensorStruct t = new cudnnTensorStruct();
            for (int i = 0; i < capacity; i++) {
                checkCudnn(cudnnCreateTensorDescriptor(t));
                this.put(i, t);
            }
            deallocator(new Deallocator(this, p));
        }

        public TensorArray(TensorArray a) {
            super(a);
        }
    }

    protected static final int TENSOR_FORMAT = CUDNN_TENSOR_NCHW;

    protected int dataType = Nd4j.dataType() == DataBuffer.Type.DOUBLE ? CUDNN_DATA_DOUBLE
                    : Nd4j.dataType() == DataBuffer.Type.FLOAT ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;
    protected int dataTypeSize =
                    Nd4j.dataType() == DataBuffer.Type.DOUBLE ? 8 : Nd4j.dataType() == DataBuffer.Type.FLOAT ? 4 : 2;
    // both CUDNN_DATA_HALF and CUDNN_DATA_FLOAT need a float value for alpha and beta
    protected Pointer alpha = dataType == CUDNN_DATA_DOUBLE ? new DoublePointer(1.0) : new FloatPointer(1.0f);
    protected Pointer beta = dataType == CUDNN_DATA_DOUBLE ? new DoublePointer(0.0) : new FloatPointer(0.0f);
    protected SizeTPointer sizeInBytes = new SizeTPointer(1);

    public boolean checkSupported() {
        // add general checks here, if any
        return true;
    }


    /**
     * From CuDNN documentation -
     * "Tensors are restricted to having at least 4 dimensions... When working with lower dimensional data, it is
     * recommended that the user create a 4Dtensor, and set the size along unused dimensions to 1."
     *
     * This method implements that - basically appends 1s to the end (shape or stride) to make it length 4,
     * or leaves it unmodified if the length is already 4 or more.
     * This method can be used for both shape and strides
     *
     * @param shapeOrStrides
     * @return
     */
    protected static int[] adaptForTensorDescr(int[] shapeOrStrides){
        if(shapeOrStrides.length >= 4)
            return shapeOrStrides;
        int[] out = new int[4];
        int i=0;
        for(; i<shapeOrStrides.length; i++ ){
            out[i] = shapeOrStrides[i];
        }
        for(; i<4; i++ ){
            out[i] = 1;
        }
        return out;
    }
}
