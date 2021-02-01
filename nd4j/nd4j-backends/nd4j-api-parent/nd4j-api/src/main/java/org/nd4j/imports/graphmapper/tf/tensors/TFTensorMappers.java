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

package org.nd4j.imports.graphmapper.tf.tensors;

import org.bytedeco.javacpp.indexer.Bfloat16ArrayIndexer;
import org.bytedeco.javacpp.indexer.HalfIndexer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.util.ArrayUtil;
import org.tensorflow.framework.TensorProto;

import java.nio.*;

public class TFTensorMappers {

    private TFTensorMappers() {}


    public static TFTensorMapper<?,?> newMapper(TensorProto tp){

        switch (tp.getDtype()){
            case DT_HALF:
                return new Float16TensorMapper(tp);
            case DT_FLOAT:
                return new Float32TensorMapper(tp);
            case DT_DOUBLE:
                return new Float64TensorMapper(tp);
            case DT_BFLOAT16:
                return new BFloat16TensorMapper(tp);

            case DT_INT8:
                return new Int8TensorMapper(tp);
            case DT_INT16:
                return new Int16TensorMapper(tp);
            case DT_INT32:
                return new Int32TensorMapper(tp);
            case DT_INT64:
                return new Int64TensorMapper(tp);


            case DT_STRING:
                return new StringTensorMapper(tp);

            case DT_BOOL:
                return new BoolTensorMapper(tp);

            case DT_UINT8:
                return new UInt8TensorMapper(tp);
            case DT_UINT16:
                return new UInt16TensorMapper(tp);
            case DT_UINT32:
                return new UInt32TensorMapper(tp);
            case DT_UINT64:
                return new UInt64TensorMapper(tp);

            case DT_QINT8:
            case DT_QUINT8:
            case DT_QINT32:
            case DT_QINT16:
            case DT_QUINT16:
                throw new IllegalStateException("Unable to map quantized type: " + tp.getDtype());
            case DT_COMPLEX64:
            case DT_COMPLEX128:
                throw new IllegalStateException("Unable to map complex type: " + tp.getDtype());
            case DT_FLOAT_REF:
            case DT_DOUBLE_REF:
            case DT_INT32_REF:
            case DT_UINT8_REF:
            case DT_INT16_REF:
            case DT_INT8_REF:
            case DT_STRING_REF:
            case DT_COMPLEX64_REF:
            case DT_INT64_REF:
            case DT_BOOL_REF:
            case DT_QINT8_REF:
            case DT_QUINT8_REF:
            case DT_QINT32_REF:
            case DT_BFLOAT16_REF:
            case DT_QINT16_REF:
            case DT_QUINT16_REF:
            case DT_UINT16_REF:
            case DT_COMPLEX128_REF:
            case DT_HALF_REF:
            case DT_RESOURCE_REF:
            case DT_VARIANT_REF:
            case DT_UINT32_REF:
            case DT_UINT64_REF:
                throw new IllegalStateException("Unable to map reference type: " + tp.getDtype());
            case UNRECOGNIZED:
            case DT_RESOURCE:
            case DT_VARIANT:
            case DT_INVALID:
            default:
                throw new IllegalStateException("Unable to map type: " + tp.getDtype());
        }
    }


    public static abstract class BaseTensorMapper<T,U extends Buffer> implements TFTensorMapper<T,U> {

        protected TensorProto tfTensor;

        public BaseTensorMapper(TensorProto tensorProto){
            this.tfTensor = tensorProto;
        }

        @Override
        public DataType dataType() {
            return ArrayOptionsHelper.convertToDataType(tfTensor.getDtype());
        }

        @Override
        public long[] shape() {
            int dims = tfTensor.getTensorShape().getDimCount();
            long[] arrayShape = new long[dims];
            for (int e = 0; e < dims; e++) {
                arrayShape[e] = tfTensor.getTensorShape().getDim(e).getSize();
            }
            return arrayShape;
        }

        @Override
        public boolean isEmpty() {
            return valueSource() == ValueSource.EMPTY;
        }

        @Override
        public ValueSource valueSource() {
            if (valueCount() > 0) {
                return ValueSource.VALUE_COUNT;
            }
            if(tfTensor.getTensorContent() != null && tfTensor.getTensorContent().size() > 0){
                return ValueSource.BINARY;
            }

            return ValueSource.EMPTY;
        }

        @Override
        public INDArray toNDArray() {
            DataType dt = dataType();
            ValueSource vs = valueSource();
            long[] shape = shape();

            INDArray out;
            switch (vs){
                case EMPTY:
                    out = Nd4j.create(dt, shape);
                    break;
                case VALUE_COUNT:
                    int n = valueCount();
                    T array = newArray(n);
                    for( int i = 0; i < n; i++) {
                        getValue(array, i);
                    }
                    out = arrayFor(shape, array);
                    break;
                case BINARY:
                    U buffer = getBuffer(tfTensor.getTensorContent().asReadOnlyByteBuffer().order(ByteOrder.nativeOrder()));
                    int m = buffer.capacity();
                    T array2 = newArray(m);
                    for( int i=0; i<m; i++ ){
                        getValue(array2, buffer, i);
                    }
                    out = arrayFor(shape, array2);
                    break;
                default:
                    throw new RuntimeException("Error converting TF tensor to INDArray");
            }

            return out;
        }
    }

    public static class Float16TensorMapper extends BaseTensorMapper<float[], Buffer>  {
        public Float16TensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            return tfTensor.getHalfValCount();
        }

        @Override
        public float[] newArray(int length) {
            return new float[length];
        }

        @Override
        public Buffer getBuffer(ByteBuffer bb) {
            throw new UnsupportedOperationException("Not yet implemnted: FP16 reading from buffer");
        }

        @Override
        public void getValue(float[] jArr, int i) {
            int asIntBytes = tfTensor.getHalfVal(i);
            jArr[i] = HalfIndexer.toFloat(asIntBytes);
        }

        @Override
        public void getValue(float[] jArr, Buffer buffer, int i){
            throw new UnsupportedOperationException("Not yet implemented: FP16 reading from buffer");
        }

        @Override
        public INDArray arrayFor(long[] shape, float[] jArr) {
            //Edge case: sometimes tf has single float value for entire array (getFloatValCount() == 1)
            if(jArr.length == 1 && ArrayUtil.prod(shape) > 1)
                return Nd4j.createUninitialized(DataType.HALF, shape).assign(jArr[0]);
            return Nd4j.create(jArr, shape, 'c').castTo(DataType.HALF);
        }
    }

    public static class Float32TensorMapper extends BaseTensorMapper<float[], FloatBuffer>  {
        public Float32TensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            return tfTensor.getFloatValCount();
        }

        @Override
        public float[] newArray(int length) {
            return new float[length];
        }

        @Override
        public FloatBuffer getBuffer(ByteBuffer bb) {
            return bb.asFloatBuffer();
        }

        @Override
        public void getValue(float[] jArr, int i) {
            jArr[i] = tfTensor.getFloatVal(i);
        }

        @Override
        public void getValue(float[] jArr, FloatBuffer buffer, int i){
            jArr[i] = buffer.get(i);
        }

        @Override
        public INDArray arrayFor(long[] shape, float[] jArr) {
            //Edge case: sometimes tf has single float value for entire array (getFloatValCount() == 1)
            if(jArr.length == 1 && ArrayUtil.prod(shape) > 1)
                return Nd4j.valueArrayOf(shape, jArr[0]);
            return Nd4j.create(jArr, shape, 'c');
        }
    }

    public static class Float64TensorMapper extends BaseTensorMapper<double[], DoubleBuffer>  {
        public Float64TensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            return tfTensor.getDoubleValCount();
        }

        @Override
        public double[] newArray(int length) {
            return new double[length];
        }

        @Override
        public DoubleBuffer getBuffer(ByteBuffer bb) {
            return bb.asDoubleBuffer();
        }

        @Override
        public void getValue(double[] jArr, int i) {
            jArr[i] = tfTensor.getDoubleVal(i);
        }

        @Override
        public void getValue(double[] jArr, DoubleBuffer buffer, int i) {
            jArr[i] = buffer.get(i);
        }

        @Override
        public INDArray arrayFor(long[] shape, double[] jArr) {
            //Edge case: sometimes tf has double float value for entire array (getDoubleValCount() == 1)
            if(jArr.length == 1 && ArrayUtil.prod(shape) > 1)
                return Nd4j.valueArrayOf(shape, jArr[0]);
            return Nd4j.create(jArr, shape, 'c');
        }
    }

    public static class BFloat16TensorMapper extends BaseTensorMapper<float[], ShortBuffer>  {
        public BFloat16TensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            return tfTensor.getHalfValCount();
        }

        @Override
        public float[] newArray(int length) {
            return new float[length];
        }

        @Override
        public ShortBuffer getBuffer(ByteBuffer bb) {
            return bb.asShortBuffer();
        }

        @Override
        public void getValue(float[] jArr, int i) {
            int asIntBytes = tfTensor.getHalfVal(i);
            jArr[i] = Bfloat16ArrayIndexer.toFloat(asIntBytes);
        }

        @Override
        public void getValue(float[] jArr, ShortBuffer buffer, int i){
            throw new UnsupportedOperationException("Not yet implemnted: BFP16 reading from buffer");
        }

        @Override
        public INDArray arrayFor(long[] shape, float[] jArr) {
            //Edge case: sometimes tf has single float value for entire array (getFloatValCount() == 1)
            if(jArr.length == 1 && ArrayUtil.prod(shape) > 1)
                return Nd4j.createUninitialized(DataType.HALF, shape).assign(jArr[0]);
            return Nd4j.create(jArr, shape, 'c').castTo(DataType.BFLOAT16);
        }
    }

    //Note TF stortes bytes as integer (other than when in a biffer)
    public static class Int8TensorMapper extends BaseTensorMapper<int[], ByteBuffer> {

        public Int8TensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            //int8 as integer
            return tfTensor.getIntValCount();
        }

        @Override
        public int[] newArray(int length) {
            return new int[length];
        }

        @Override
        public ByteBuffer getBuffer(ByteBuffer bb) {
            return bb;
        }

        @Override
        public void getValue(int[] jArr, int i) {
            jArr[i] = tfTensor.getIntVal(i);
        }

        @Override
        public void getValue(int[] jArr, ByteBuffer buffer, int i) {
            jArr[i] = buffer.get(i);
        }

        @Override
        public INDArray arrayFor(long[] shape, int[] jArr) {
            DataType dt = dataType();
            return Nd4j.create(Nd4j.createTypedBuffer(jArr, dt), shape,Nd4j.getStrides(shape, 'c'), 0, 'c', dt);
        }
    }

    public static class Int16TensorMapper extends BaseTensorMapper<int[], ShortBuffer> {

        public Int16TensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            //Shorts as integer
            return tfTensor.getIntValCount();
        }

        @Override
        public int[] newArray(int length) {
            return new int[length];
        }

        @Override
        public ShortBuffer getBuffer(ByteBuffer bb) {
            return bb.asShortBuffer();
        }

        @Override
        public void getValue(int[] jArr, int i) {
            jArr[i] = tfTensor.getIntVal(i);
        }

        @Override
        public void getValue(int[] jArr, ShortBuffer buffer, int i) {
            jArr[i] = buffer.get(i);
        }

        @Override
        public INDArray arrayFor(long[] shape, int[] jArr) {
            DataType dt = dataType();
            return Nd4j.create(Nd4j.createTypedBuffer(jArr, dt), shape,Nd4j.getStrides(shape, 'c'), 0, 'c', dt);
        }
    }


    public static class Int32TensorMapper extends BaseTensorMapper<int[], IntBuffer> {

        public Int32TensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            return tfTensor.getIntValCount();
        }

        @Override
        public int[] newArray(int length) {
            return new int[length];
        }

        @Override
        public IntBuffer getBuffer(ByteBuffer bb) {
            return bb.asIntBuffer();
        }

        @Override
        public void getValue(int[] jArr, int i) {
            jArr[i] = tfTensor.getIntVal(i);
        }

        @Override
        public void getValue(int[] jArr, IntBuffer buffer, int i) {
            jArr[i] = buffer.get(i);
        }

        @Override
        public INDArray arrayFor(long[] shape, int[] jArr) {
            DataType dt = dataType();
            return Nd4j.create(Nd4j.createTypedBuffer(jArr, dt), shape,Nd4j.getStrides(shape, 'c'), 0, 'c', dt);
        }
    }

    public static class Int64TensorMapper extends BaseTensorMapper<long[], LongBuffer> {

        public Int64TensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            return tfTensor.getInt64ValCount();
        }

        @Override
        public long[] newArray(int length) {
            return new long[length];
        }

        @Override
        public LongBuffer getBuffer(ByteBuffer bb) {
            return bb.asLongBuffer();
        }

        @Override
        public void getValue(long[] jArr, int i) {
            jArr[i] = tfTensor.getInt64Val(i);
        }

        @Override
        public void getValue(long[] jArr, LongBuffer buffer, int i) {
            jArr[i] = buffer.get(i);
        }

        @Override
        public INDArray arrayFor(long[] shape, long[] jArr) {
            DataType dt = dataType();
            return Nd4j.create(Nd4j.createTypedBuffer(jArr, dt), shape,Nd4j.getStrides(shape, 'c'), 0, 'c', dt);
        }
    }

    //Note TF stortes bytes as integer (other than when in a buffer)
    public static class UInt8TensorMapper extends BaseTensorMapper<int[], ByteBuffer> {

        public UInt8TensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            //int8 as integer
            return tfTensor.getIntValCount();
        }

        @Override
        public int[] newArray(int length) {
            return new int[length];
        }

        @Override
        public ByteBuffer getBuffer(ByteBuffer bb) {
            return bb;
        }

        @Override
        public void getValue(int[] jArr, int i) {
            jArr[i] = tfTensor.getIntVal(i);
        }

        @Override
        public void getValue(int[] jArr, ByteBuffer buffer, int i) {
            byte b = buffer.get(i); //Signed, but bytes are really for unsigned...
            jArr[i] = b & 0xff;
        }

        @Override
        public INDArray arrayFor(long[] shape, int[] jArr) {
            DataType dt = dataType();
            return Nd4j.create(Nd4j.createTypedBuffer(jArr, dt), shape,Nd4j.getStrides(shape, 'c'), 0, 'c', dt);
        }
    }

    public static class UInt16TensorMapper extends BaseTensorMapper<int[], ShortBuffer> {

        public UInt16TensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            //int8 as integer
            return tfTensor.getIntValCount();
        }

        @Override
        public int[] newArray(int length) {
            return new int[length];
        }

        @Override
        public ShortBuffer getBuffer(ByteBuffer bb) {
            return bb.asShortBuffer();
        }

        @Override
        public void getValue(int[] jArr, int i) {
            jArr[i] = tfTensor.getIntVal(i);
        }

        @Override
        public void getValue(int[] jArr, ShortBuffer buffer, int i) {
            short b = buffer.get(i); //Signed, but bytes are really for unsigned...
            jArr[i] = b & 0xffff;
        }

        @Override
        public INDArray arrayFor(long[] shape, int[] jArr) {
            DataType dt = dataType();
            return Nd4j.create(Nd4j.createTypedBuffer(jArr, dt), shape,Nd4j.getStrides(shape, 'c'), 0, 'c', dt);
        }
    }

    public static class UInt32TensorMapper extends BaseTensorMapper<long[], IntBuffer> {

        public UInt32TensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            //int8 as integer
            return tfTensor.getInt64ValCount();
        }

        @Override
        public long[] newArray(int length) {
            return new long[length];
        }

        @Override
        public IntBuffer getBuffer(ByteBuffer bb) {
            return bb.asIntBuffer();
        }

        @Override
        public void getValue(long[] jArr, int i) {
            jArr[i] = tfTensor.getInt64Val(i);
        }

        @Override
        public void getValue(long[] jArr, IntBuffer buffer, int i) {
            int b = buffer.get(i); //Signed, but bytes are really for unsigned...
            jArr[i] = b & 0xffffffffL;
        }

        @Override
        public INDArray arrayFor(long[] shape, long[] jArr) {
            DataType dt = dataType();
            return Nd4j.create(Nd4j.createTypedBuffer(jArr, dt), shape,Nd4j.getStrides(shape, 'c'), 0, 'c', dt);
        }
    }

    public static class UInt64TensorMapper extends BaseTensorMapper<long[], LongBuffer> {

        public UInt64TensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            //int8 as integer
            return tfTensor.getInt64ValCount();
        }

        @Override
        public long[] newArray(int length) {
            return new long[length];
        }

        @Override
        public LongBuffer getBuffer(ByteBuffer bb) {
            return bb.asLongBuffer();
        }

        @Override
        public void getValue(long[] jArr, int i) {
            //TODO out of range for largest values!
            jArr[i] = tfTensor.getInt64Val(i);
        }

        @Override
        public void getValue(long[] jArr, LongBuffer buffer, int i) {
            //TODO out of range for largest values!
            jArr[i] = buffer.get(i);
        }

        @Override
        public INDArray arrayFor(long[] shape, long[] jArr) {
            DataType dt = dataType();
            return Nd4j.create(Nd4j.createTypedBuffer(jArr, dt), shape,Nd4j.getStrides(shape, 'c'), 0, 'c', dt);
        }
    }


    public static class StringTensorMapper extends BaseTensorMapper<String[], ByteBuffer> {
        public StringTensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            return tfTensor.getStringValCount();
        }

        @Override
        public String[] newArray(int length) {
            return new String[length];
        }

        @Override
        public ByteBuffer getBuffer(ByteBuffer bb) {
            throw new UnsupportedOperationException("Not supported for String types");
        }

        @Override
        public void getValue(String[] jArr, int i) {
            jArr[i] = tfTensor.getStringVal(i).toStringUtf8();
        }

        @Override
        public void getValue(String[] jArr, ByteBuffer buffer, int i) {
            throw new UnsupportedOperationException("Not supported for String types");
        }

        @Override
        public INDArray arrayFor(long[] shape, String[] jArr) {
            return Nd4j.create(jArr).reshape(shape);
        }
    }

    public static class BoolTensorMapper extends BaseTensorMapper<boolean[], ByteBuffer> {
        public BoolTensorMapper(TensorProto tensorProto) {
            super(tensorProto);
        }

        @Override
        public int valueCount() {
            return tfTensor.getBoolValCount();
        }

        @Override
        public boolean[] newArray(int length) {
            return new boolean[length];
        }

        @Override
        public ByteBuffer getBuffer(ByteBuffer bb) {
            throw new UnsupportedOperationException("Not supported for String types");
        }

        @Override
        public void getValue(boolean[] jArr, int i) {
            jArr[i] = tfTensor.getBoolVal(i);
        }

        @Override
        public void getValue(boolean[] jArr, ByteBuffer buffer, int i) {
            throw new UnsupportedOperationException("Not supported for boolean types");
        }

        @Override
        public INDArray arrayFor(long[] shape, boolean[] jArr) {
            return Nd4j.create(jArr).reshape(shape);
        }
    }
}
