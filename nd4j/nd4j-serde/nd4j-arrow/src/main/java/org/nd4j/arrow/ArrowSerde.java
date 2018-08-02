/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.arrow;

import com.google.flatbuffers.FlatBufferBuilder;
import org.apache.arrow.flatbuf.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Conversion to and from arrow {@link Tensor}
 * and {@link INDArray}
 *
 * @author Adam Gibson
 */
public class ArrowSerde {


    /**
     * Convert a {@link Tensor}
     * to an {@link INDArray}
     * @param tensor the input tensor
     * @return the equivalent {@link INDArray}
     */
    public static INDArray fromTensor(Tensor tensor) {
        byte b = tensor.typeType();
        int[] shape = new int[tensor.shapeLength()];
        int[] stride = new int[tensor.stridesLength()];
        for(int i = 0; i < shape.length; i++) {
            shape[i] = (int) tensor.shape(i).size();
            stride[i] = (int) tensor.strides(i);
        }

        int length = ArrayUtil.prod(shape);
        Buffer buffer = tensor.data();
        if(buffer == null) {
            throw new ND4JIllegalStateException("Buffer was not serialized properly.");
        }
        //deduce element size
        int elementSize = (int) buffer.length() / length;
        //nd4j strides aren't  based on element size
        for(int i = 0; i < stride.length; i++) {
            stride[i] /= elementSize;
        }

        DataBuffer.Type  type = typeFromTensorType(b,elementSize);
        DataBuffer dataBuffer = DataBufferStruct.createFromByteBuffer(tensor.getByteBuffer(),(int) tensor.data().offset(),type,length);
        INDArray arr = Nd4j.create(dataBuffer,shape);
        arr.setShapeAndStride(shape,stride);
        return arr;
    }

    /**
     * Convert an {@link INDArray}
     * to an arrow {@link Tensor}
     * @param arr the array to convert
     * @return the equivalent {@link Tensor}
     */
    public static Tensor toTensor(INDArray arr) {
        FlatBufferBuilder bufferBuilder = new FlatBufferBuilder(1024);
        long[] strides = getArrowStrides(arr);
        int shapeOffset = createDims(bufferBuilder,arr);
        int stridesOffset = Tensor.createStridesVector(bufferBuilder,strides);

        Tensor.startTensor(bufferBuilder);

        addTypeTypeRelativeToNDArray(bufferBuilder,arr);
        Tensor.addShape(bufferBuilder,shapeOffset);
        Tensor.addStrides(bufferBuilder,stridesOffset);

        Tensor.addData(bufferBuilder,addDataForArr(bufferBuilder,arr));
        int endTensor = Tensor.endTensor(bufferBuilder);
        Tensor.finishTensorBuffer(bufferBuilder,endTensor);
        return Tensor.getRootAsTensor(bufferBuilder.dataBuffer());
    }


    /**
     * Create a {@link Buffer}
     * representing the location metadata of the actual data
     * contents for the ndarrays' {@link DataBuffer}
     * @param bufferBuilder the buffer builder in use
     * @param arr the array to add the underlying data for
     * @return the offset added
     */
    public static int addDataForArr(FlatBufferBuilder bufferBuilder, INDArray arr) {
        DataBuffer toAdd = arr.isView() ? arr.dup().data() : arr.data();
        int offset = DataBufferStruct.createDataBufferStruct(bufferBuilder,toAdd);
        int ret = Buffer.createBuffer(bufferBuilder,offset,toAdd.length() * toAdd.getElementSize());
        return ret;

    }

    /**
     * Convert the given {@link INDArray}
     * data  type to the proper data type for the tensor.
     * @param bufferBuilder the buffer builder in use
     * @param arr the array to conver tthe data type for
     */
    public static void addTypeTypeRelativeToNDArray(FlatBufferBuilder bufferBuilder,INDArray arr) {
        switch(arr.data().dataType()) {
            case LONG:
            case INT:
                Tensor.addTypeType(bufferBuilder,Type.Int);
                break;
            case FLOAT:
                Tensor.addTypeType(bufferBuilder,Type.FloatingPoint);
                break;
            case DOUBLE:
                Tensor.addTypeType(bufferBuilder,Type.Decimal);
                break;
        }
    }

    /**
     * Create the dimensions for the flatbuffer builder
     * @param bufferBuilder the buffer builder to use
     * @param arr the input array
     * @return
     */
    public static int createDims(FlatBufferBuilder bufferBuilder,INDArray arr) {
        int[] tensorDimOffsets = new int[arr.rank()];
        int[] nameOffset = new int[arr.rank()];
        for(int i = 0; i < tensorDimOffsets.length; i++) {
            nameOffset[i] = bufferBuilder.createString("");
            tensorDimOffsets[i] = TensorDim.createTensorDim(bufferBuilder,arr.size(i),nameOffset[i]);
        }

        return Tensor.createShapeVector(bufferBuilder,tensorDimOffsets);
    }


    /**
     * Get the strides of this {@link INDArray}
     * multiplieed by  the element size.
     * This is the {@link Tensor} and numpy format
     * @param arr the array to convert
     * @return
     */
    public static long[] getArrowStrides(INDArray arr) {
        long[] ret = new long[arr.rank()];
        for(int i = 0; i < arr.rank(); i++) {
            ret[i] = arr.stride(i) * arr.data().getElementSize();
        }

        return ret;
    }



    /**
     * Create thee databuffer type frm the given type,
     * relative to the bytes in arrow in class:
     * {@link Type}
     * @param type the type to create the nd4j {@link DataBuffer.Type} from
     * @param elementSize the element size
     * @return the data buffer type
     */
    public static DataBuffer.Type typeFromTensorType(byte type,int elementSize) {
        if(type == Type.FloatingPoint) {
            return DataBuffer.Type.FLOAT;
        }
        else if(type == Type.Decimal) {
            return DataBuffer.Type.DOUBLE;
        }
        else if(type == Type.Int) {
            if(elementSize == 4) {
                return DataBuffer.Type.INT;
            }
            else if(elementSize == 8) {
                return DataBuffer.Type.LONG;
            }
        }
        else {
            throw new IllegalArgumentException("Only valid types are Type.Decimal and Type.Int");
        }

        throw new IllegalArgumentException("Unable to determine data type");
    }
}
