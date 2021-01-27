/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.python;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.apache.commons.lang3.ArrayUtils;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.nd4j.linalg.api.buffer.DataType.FLOAT;


/**
 * Wrapper around INDArray for initializing from numpy array
 *
 * @author Fariz Rahman
 */
@Getter
@NoArgsConstructor
public class NumpyArray {

    private static NativeOps nativeOps;
    private static Map<String, INDArray> arrayCache;  // Avoids re-allocation of device buffer
    private long address;
    private long[] shape;
    private long[] strides;
    private DataType dtype;
    private INDArray nd4jArray;

    static {
        //initialize
        Nd4j.scalar(1.0);
        nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        arrayCache = new HashMap<>();
    }

    @Builder
    public NumpyArray(long address, long[] shape, long strides[], DataType dtype, boolean copy) {
        this.address = address;
        this.shape = shape;
        this.strides = strides;
        this.dtype = dtype;
        setND4JArray();
        if (copy) {
            nd4jArray = nd4jArray.dup();
            Nd4j.getAffinityManager().ensureLocation(nd4jArray, AffinityManager.Location.HOST);
            this.address = nd4jArray.data().address();
        }
    }



    public NumpyArray copy() {
        return new NumpyArray(nd4jArray.dup());
    }

    public NumpyArray(long address, long[] shape, long strides[]) {
        this(address, shape, strides, FLOAT, false);
    }

    public NumpyArray(long address, long[] shape, long strides[], DataType dtype) {
        this(address, shape, strides, dtype, false);
    }


    private void setND4JArray() {

        long size = 1;
        for (long d : shape) {
            size *= d;
        }

        String cacheKey = address + "_" + size + "_" + dtype + "_" + ArrayUtils.toString(strides);
        nd4jArray = arrayCache.get(cacheKey);
        if (nd4jArray == null) {
            Pointer ptr = nativeOps.pointerForAddress(address);
            ptr = ptr.limit(size);
            ptr = ptr.capacity(size);
            DataBuffer buff = Nd4j.createBuffer(ptr, size, dtype);

            int elemSize = buff.getElementSize();
            long[] nd4jStrides = new long[strides.length];
            for (int i = 0; i < strides.length; i++) {
                nd4jStrides[i] = strides[i] / elemSize;
            }

            nd4jArray = Nd4j.create(buff, shape, nd4jStrides, 0, Shape.getOrder(shape, nd4jStrides, 1), dtype);
            arrayCache.put(cacheKey, nd4jArray);
        }
        else{
            if (!Arrays.equals(nd4jArray.shape(), shape)){
                nd4jArray = nd4jArray.reshape(shape);
            }
        }
        Nd4j.getAffinityManager().ensureLocation(nd4jArray, AffinityManager.Location.HOST);
    }

    public INDArray getNd4jArray(){
        Nd4j.getAffinityManager().tagLocation(nd4jArray, AffinityManager.Location.HOST);
        return nd4jArray;
    }

    public NumpyArray(INDArray nd4jArray) {
        Nd4j.getAffinityManager().ensureLocation(nd4jArray, AffinityManager.Location.HOST);
        DataBuffer buff = nd4jArray.data();
        address = buff.pointer().address();
        shape = nd4jArray.shape();
        long[] nd4jStrides = nd4jArray.stride();
        strides = new long[nd4jStrides.length];
        int elemSize = buff.getElementSize();
        for (int i = 0; i < strides.length; i++) {
            strides[i] = nd4jStrides[i] * elemSize;
        }
        dtype = nd4jArray.dataType();
        this.nd4jArray = nd4jArray;
        String cacheKey = address + "_" + nd4jArray.length() + "_" + dtype + "_" + ArrayUtils.toString(strides);
        arrayCache.put(cacheKey, nd4jArray);
    }

}