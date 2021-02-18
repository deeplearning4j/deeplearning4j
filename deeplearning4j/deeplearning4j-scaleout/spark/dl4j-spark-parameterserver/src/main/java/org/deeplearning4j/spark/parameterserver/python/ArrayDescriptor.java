/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.spark.parameterserver.python;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;


public class ArrayDescriptor implements java.io.Serializable{

    private long address;
    private long[] shape;
    private long[] stride;
    DataType type;
    char ordering;
    private static NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();

    public ArrayDescriptor(INDArray array) throws Exception{
        this(array.data().address(), array.shape(), array.stride(), array.data().dataType(), array.ordering());
        if (array.isEmpty()){
            throw new UnsupportedOperationException("Empty arrays are not supported");
        }
    }

    public ArrayDescriptor(long address, long[] shape, long[] stride, DataType type, char ordering){
        this.address = address;
        this.shape = shape;
        this.stride = stride;
        this.type = type;
        this.ordering = ordering;
    }
    public long getAddress(){
        return address;
    }

    public long[] getShape(){
        return shape;
    }

    public long[] getStride(){
        return stride;
    }

    public DataType getType(){
        return type;
    }

    public char getOrdering(){
        return ordering;
    }

    private long size(){
        long s = 1;
        for (long d: shape){
            s *= d;
        }
        return s;
    }

    public INDArray getArray() {
        Pointer ptr = nativeOps.pointerForAddress(address);
        ptr = ptr.limit(size());
        DataBuffer buff = Nd4j.createBuffer(ptr, size(), type);
        return Nd4j.create(buff, shape, stride, 0, ordering, type);
    }

}
