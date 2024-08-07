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
package org.nd4j.linalg.factory;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

public class NDArrayCreateArgs {

    private DataBufferCreator dataBufferCreator;
    private ArrayMetaData arrayMetaData;



    public INDArray createWithData() {
        return Nd4j.create(
                dataBufferCreator.dataBuffer,
                arrayMetaData.shape,
                arrayMetaData.stride,
                arrayMetaData.offset,
                arrayMetaData.order);
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ArrayMetaData {
        private long[] shape;
        private long[] stride;
        private char order;
        private long offset;
        private boolean isView;

    }


    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class DataBufferCreator {
        private double[] doubleData;
        private double[][] doubleDataMatrix;
        private double[][][] doubleTensorMatrix;

        private float[] floatData;
        private float[][] floatDataMatrix;
        private float[][][] floatTensorMatrix;

        private int[] intData;
        private int[][] intDataMatrix;
        private int[][][] intTensorMatrix;

        private long[] longData;
        private long[][] longDataMatrix;
        private long[][][] longTensorMatrix;

        private byte[] byteData;
        private byte[][] byteDataMatrix;
        private byte[][][] byteTensorMatrix;

        private boolean[] boolData;
        private boolean[][] boolDataMatrix;
        private boolean[][][] boolTensorMatrix;

        private String[] stringData;
        private String[][] stringDataMatrix;
        private String[][][] stringDataTensor;

        private short[] shortData;
        private short[][] shortDataMatrix;
        private short[][][] shortTensorMatrix;



        private DataBuffer dataBuffer;

        @Builder.Default
        private DataType dataType = DataType.UNKNOWN;

        private ArrayMetaData.ArrayMetaDataBuilder arrayMetaDataBuilder = ArrayMetaData.builder();

        //if a data type isn't specified by the user we should infer the default type
        private DataType defaultDependingOnType() {
            if(doubleData != null || doubleDataMatrix != null || doubleTensorMatrix != null)
                return DataType.DOUBLE;
            else if(floatData != null || floatDataMatrix != null || floatTensorMatrix != null)
                return DataType.FLOAT;
            else if(intData != null || intDataMatrix != null || intTensorMatrix != null)
                return DataType.INT;
            else if(longData != null || longDataMatrix != null || longTensorMatrix != null)
                return DataType.LONG;
            else if(byteData != null || byteDataMatrix != null || byteTensorMatrix != null)
                return DataType.BYTE;
            else if(boolData != null || boolDataMatrix != null || boolTensorMatrix != null)
                return DataType.BOOL;
            else if(stringData != null || stringDataMatrix != null || stringDataTensor != null)
                return DataType.UTF8;
            else if(shortData != null || shortDataMatrix != null || shortTensorMatrix != null)
                return DataType.SHORT;
            else if(dataBuffer != null)
                return dataBuffer.dataType();
            else
                throw new IllegalStateException("Unable to infer data type");

        }

        private DataBuffer createDataBuffer() {
            if(dataType == null || dataType == DataType.UNKNOWN)
                dataType = defaultDependingOnType();
            if (doubleData != null) {
                return Nd4j.createTypedBuffer(doubleData,dataType);
            } else if (doubleDataMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(doubleDataMatrix),dataType);
            } else if (doubleTensorMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(doubleTensorMatrix),dataType);
            } else if (floatData != null) {
                return Nd4j.createTypedBuffer(floatData,dataType);
            } else if (floatDataMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(floatDataMatrix),dataType);
            } else if (floatTensorMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(floatTensorMatrix),dataType);
            } else if (intData != null) {
                return Nd4j.createTypedBuffer(intData,dataType);
            } else if (intDataMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(intDataMatrix),dataType);
            } else if (intTensorMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(intTensorMatrix),dataType);
            } else if (longData != null) {
                return Nd4j.createTypedBuffer(longData,dataType);
            } else if (longDataMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(longDataMatrix),dataType);
            } else if (longTensorMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(longTensorMatrix),dataType);
            } else if (byteData != null) {
                return Nd4j.createTypedBuffer(byteData,dataType);
            } else if (byteDataMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(byteDataMatrix),dataType);
            } else if (byteTensorMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(byteTensorMatrix),dataType);
            } else if (boolData != null) {
                return Nd4j.createTypedBuffer(boolData,dataType);
            } else if (boolDataMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(boolDataMatrix),dataType);
            } else if (boolTensorMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(boolTensorMatrix),dataType);
            } else if (stringData != null) {
                return Nd4j.createTypedBuffer(stringData,dataType);
            } else if (stringDataMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(stringDataMatrix),dataType);
            } else if (stringDataTensor != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(stringDataTensor),dataType);
            } else if (shortData != null) {
                return Nd4j.createTypedBuffer(shortData,dataType);
            } else if (shortDataMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(shortDataMatrix),dataType);
            } else if (shortTensorMatrix != null) {
                return Nd4j.createTypedBuffer(ArrayUtil.flatten(shortTensorMatrix),dataType);
            } else if (dataBuffer != null) {
                return dataBuffer;
            }

            throw new IllegalStateException("Data buffer was not created! Set a data source (an array or data buffer)");
        }

        private DataBufferCreator create() {
            dataBuffer = createDataBuffer();
            return this;
        }

        public ArrayMetaData.ArrayMetaDataBuilder buildArray() {
            return arrayMetaDataBuilder;
        }
        

    }


}
