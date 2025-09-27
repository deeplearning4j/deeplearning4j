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

package org.nd4j.onnx;

import onnx.Onnx;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.protobuf.ByteString;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Utility class for converting between ONNX TensorProto and ND4J INDArray objects.
 *
 * This class provides methods to convert ONNX tensors to ND4J arrays and vice versa,
 * handling various data types and memory layouts.
 *
 * @author Adam Gibson
 */
public class OnnxTensorUtils {

    /**
     * Convert an ONNX TensorProto to an ND4J INDArray.
     *
     * @param tensorProto the ONNX tensor to convert
     * @return the converted INDArray
     * @throws IllegalArgumentException if the tensor data type is not supported
     */
    public static INDArray toINDArray(Onnx.TensorProto tensorProto) {
        if (tensorProto == null) {
            throw new IllegalArgumentException("TensorProto cannot be null");
        }

        // Extract shape
        long[] shape = tensorProto.getDimsList().stream().mapToLong(Long::longValue).toArray();

        // Handle scalar case
        if (shape.length == 0) {
            shape = new long[]{1};
        }

        // Convert data type and create array
        Onnx.TensorProto.DataType onnxDataType = Onnx.TensorProto.DataType.forNumber(tensorProto.getDataType());
        DataType dataType = onnxDataTypeToNd4j(onnxDataType);
        INDArray result;

        switch (onnxDataType) {
            case FLOAT:
                result = createFloatArray(tensorProto, shape, dataType);
                break;
            case DOUBLE:
                result = createDoubleArray(tensorProto, shape, dataType);
                break;
            case INT32:
                result = createInt32Array(tensorProto, shape, dataType);
                break;
            case INT64:
                result = createInt64Array(tensorProto, shape, dataType);
                break;
            case UINT32:
                result = createUInt32Array(tensorProto, shape, dataType);
                break;
            case UINT64:
                result = createUInt64Array(tensorProto, shape, dataType);
                break;
            case INT16:
                result = createInt16Array(tensorProto, shape, dataType);
                break;
            case UINT16:
                result = createUInt16Array(tensorProto, shape, dataType);
                break;
            case INT8:
            case UINT8:
                result = createByteArray(tensorProto, shape, dataType);
                break;
            case BOOL:
                result = createBoolArray(tensorProto, shape, dataType);
                break;
            case FLOAT16:
                result = createFloat16Array(tensorProto, shape, dataType);
                break;
            case BFLOAT16:
                result = createBFloat16Array(tensorProto, shape, dataType);
                break;
            default:
                throw new IllegalArgumentException("Unsupported ONNX data type: " + onnxDataType);
        }

        return result;
    }

    /**
     * Convert an ND4J INDArray to an ONNX TensorProto.
     *
     * @param array the INDArray to convert
     * @param name the name for the tensor (optional, can be null)
     * @return the converted ONNX TensorProto
     * @throws IllegalArgumentException if the array data type is not supported
     */
    public static Onnx.TensorProto toTensorProto(INDArray array, String name) {
        if (array == null) {
            throw new IllegalArgumentException("INDArray cannot be null");
        }

        Onnx.TensorProto.Builder builder = Onnx.TensorProto.newBuilder();

        if (name != null) {
            builder.setName(name);
        }

        // Set dimensions
        for (long dim : array.shape()) {
            builder.addDims(dim);
        }

        // Set data type and data
        DataType nd4jType = array.dataType();
        Onnx.TensorProto.DataType onnxType = nd4jDataTypeToOnnx(nd4jType);
        builder.setDataType(onnxType.getNumber());

        switch (nd4jType) {
            case FLOAT:
                addFloatData(builder, array);
                break;
            case DOUBLE:
                addDoubleData(builder, array);
                break;
            case INT:  // ND4J deprecated INT, equivalent to INT32
                addInt32Data(builder, array);
                break;
            case LONG: // ND4J deprecated LONG, equivalent to INT64
                addInt64Data(builder, array);
                break;
            case SHORT: // ND4J deprecated SHORT, equivalent to INT16
                addInt16Data(builder, array);
                break;
            case BYTE: // ND4J deprecated BYTE, equivalent to INT8
            case UBYTE: // ND4J deprecated UBYTE, equivalent to UINT8
                addByteData(builder, array);
                break;
            case BOOL:
                addBoolData(builder, array);
                break;
            case HALF: // ND4J deprecated HALF, equivalent to FLOAT16
                addFloat16Data(builder, array);
                break;
            case BFLOAT16:
                addBFloat16Data(builder, array);
                break;
            case UINT16:
                addUInt16Data(builder, array);
                break;
            case UINT32:
                addUInt32Data(builder, array);
                break;
            case UINT64:
                addUInt64Data(builder, array);
                break;
            default:
                throw new IllegalArgumentException("Unsupported ND4J data type: " + nd4jType);
        }

        return builder.build();
    }

    /**
     * Convert an ND4J INDArray to an ONNX TensorProto with no name.
     *
     * @param array the INDArray to convert
     * @return the converted ONNX TensorProto
     */
    public static Onnx.TensorProto toTensorProto(INDArray array) {
        return toTensorProto(array, null);
    }

    // Helper methods for creating arrays from ONNX data

    private static INDArray createFloatArray(Onnx.TensorProto tensorProto, long[] shape, DataType dataType) {
        if (tensorProto.getFloatDataCount() > 0) {
            float[] data = new float[tensorProto.getFloatDataCount()];
            for (int i = 0; i < data.length; i++) {
                data[i] = tensorProto.getFloatData(i);
            }
            return Nd4j.createFromArray(data).reshape(shape);
        } else if (tensorProto.getRawData() != null && !tensorProto.getRawData().isEmpty()) {
            return createFromRawData(tensorProto.getRawData(), shape, dataType);
        } else {
            return Nd4j.zeros(dataType, shape);
        }
    }

    private static INDArray createDoubleArray(Onnx.TensorProto tensorProto, long[] shape, DataType dataType) {
        if (tensorProto.getDoubleDataCount() > 0) {
            double[] data = new double[tensorProto.getDoubleDataCount()];
            for (int i = 0; i < data.length; i++) {
                data[i] = tensorProto.getDoubleData(i);
            }
            return Nd4j.createFromArray(data).reshape(shape);
        } else if (tensorProto.getRawData() != null && !tensorProto.getRawData().isEmpty()) {
            return createFromRawData(tensorProto.getRawData(), shape, dataType);
        } else {
            return Nd4j.zeros(dataType, shape);
        }
    }

    private static INDArray createInt32Array(Onnx.TensorProto tensorProto, long[] shape, DataType dataType) {
        if (tensorProto.getInt32DataCount() > 0) {
            int[] data = new int[tensorProto.getInt32DataCount()];
            for (int i = 0; i < data.length; i++) {
                data[i] = tensorProto.getInt32Data(i);
            }
            return Nd4j.createFromArray(data).reshape(shape);
        } else if (tensorProto.getRawData() != null && !tensorProto.getRawData().isEmpty()) {
            return createFromRawData(tensorProto.getRawData(), shape, dataType);
        } else {
            return Nd4j.zeros(dataType, shape);
        }
    }

    private static INDArray createInt64Array(Onnx.TensorProto tensorProto, long[] shape, DataType dataType) {
        if (tensorProto.getInt64DataCount() > 0) {
            long[] data = new long[tensorProto.getInt64DataCount()];
            for (int i = 0; i < data.length; i++) {
                data[i] = tensorProto.getInt64Data(i);
            }
            return Nd4j.createFromArray(data).reshape(shape);
        } else if (tensorProto.getRawData() != null && !tensorProto.getRawData().isEmpty()) {
            return createFromRawData(tensorProto.getRawData(), shape, dataType);
        } else {
            return Nd4j.zeros(dataType, shape);
        }
    }

    private static INDArray createUInt32Array(Onnx.TensorProto tensorProto, long[] shape, DataType dataType) {
        if (tensorProto.getUint64DataCount() > 0) {
            // ONNX uses uint64Data for uint32 values
            long[] data = new long[tensorProto.getUint64DataCount()];
            for (int i = 0; i < data.length; i++) {
                data[i] = tensorProto.getUint64Data(i);
            }
            return Nd4j.createFromArray(data).reshape(shape);
        } else if (tensorProto.getRawData() != null && !tensorProto.getRawData().isEmpty()) {
            return createFromRawData(tensorProto.getRawData(), shape, dataType);
        } else {
            return Nd4j.zeros(dataType, shape);
        }
    }

    private static INDArray createUInt64Array(Onnx.TensorProto tensorProto, long[] shape, DataType dataType) {
        if (tensorProto.getUint64DataCount() > 0) {
            long[] data = new long[tensorProto.getUint64DataCount()];
            for (int i = 0; i < data.length; i++) {
                data[i] = tensorProto.getUint64Data(i);
            }
            return Nd4j.createFromArray(data).reshape(shape);
        } else if (tensorProto.getRawData() != null && !tensorProto.getRawData().isEmpty()) {
            return createFromRawData(tensorProto.getRawData(), shape, dataType);
        } else {
            return Nd4j.zeros(dataType, shape);
        }
    }

    private static INDArray createInt16Array(Onnx.TensorProto tensorProto, long[] shape, DataType dataType) {
        if (tensorProto.getInt32DataCount() > 0) {
            // ONNX uses int32Data for int16 values
            short[] data = new short[tensorProto.getInt32DataCount()];
            for (int i = 0; i < data.length; i++) {
                data[i] = (short) tensorProto.getInt32Data(i);
            }
            return Nd4j.createFromArray(data).reshape(shape);
        } else if (tensorProto.getRawData() != null && !tensorProto.getRawData().isEmpty()) {
            return createFromRawData(tensorProto.getRawData(), shape, dataType);
        } else {
            return Nd4j.zeros(dataType, shape);
        }
    }

    private static INDArray createUInt16Array(Onnx.TensorProto tensorProto, long[] shape, DataType dataType) {
        if (tensorProto.getInt32DataCount() > 0) {
            // ONNX uses int32Data for uint16 values, map to INT32 in ND4J
            int[] data = new int[tensorProto.getInt32DataCount()];
            for (int i = 0; i < data.length; i++) {
                data[i] = tensorProto.getInt32Data(i) & 0xFFFF; // Mask to unsigned 16-bit
            }
            return Nd4j.createFromArray(data).reshape(shape);
        } else if (tensorProto.getRawData() != null && !tensorProto.getRawData().isEmpty()) {
            return createFromRawData(tensorProto.getRawData(), shape, dataType);
        } else {
            return Nd4j.zeros(dataType, shape);
        }
    }

    private static INDArray createByteArray(Onnx.TensorProto tensorProto, long[] shape, DataType dataType) {
        if (tensorProto.getRawData() != null && !tensorProto.getRawData().isEmpty()) {
            return createFromRawData(tensorProto.getRawData(), shape, dataType);
        } else {
            return Nd4j.zeros(dataType, shape);
        }
    }

    private static INDArray createBoolArray(Onnx.TensorProto tensorProto, long[] shape, DataType dataType) {
        if (tensorProto.getInt32DataCount() > 0) {
            // ONNX uses int32Data for bool values
            boolean[] data = new boolean[tensorProto.getInt32DataCount()];
            for (int i = 0; i < data.length; i++) {
                data[i] = tensorProto.getInt32Data(i) != 0;
            }
            return Nd4j.createFromArray(data).reshape(shape);
        } else if (tensorProto.getRawData() != null && !tensorProto.getRawData().isEmpty()) {
            return createFromRawData(tensorProto.getRawData(), shape, dataType);
        } else {
            return Nd4j.zeros(dataType, shape);
        }
    }

    private static INDArray createFloat16Array(Onnx.TensorProto tensorProto, long[] shape, DataType dataType) {
        if (tensorProto.getInt32DataCount() > 0) {
            // ONNX uses int32Data for float16 values (as 16-bit integers)
            float[] data = new float[tensorProto.getInt32DataCount()];
            for (int i = 0; i < data.length; i++) {
                // Convert from half precision to float
                data[i] = Float.intBitsToFloat(halfToFloat(tensorProto.getInt32Data(i) & 0xFFFF));
            }
            return Nd4j.createFromArray(data).reshape(shape);
        } else if (tensorProto.getRawData() != null && !tensorProto.getRawData().isEmpty()) {
            return createFromRawData(tensorProto.getRawData(), shape, dataType);
        } else {
            return Nd4j.zeros(dataType, shape);
        }
    }

    private static INDArray createBFloat16Array(Onnx.TensorProto tensorProto, long[] shape, DataType dataType) {
        if (tensorProto.getInt32DataCount() > 0) {
            // ONNX uses int32Data for bfloat16 values (as 16-bit integers)
            float[] data = new float[tensorProto.getInt32DataCount()];
            for (int i = 0; i < data.length; i++) {
                // Convert from bfloat16 to float
                int bfloat16Bits = tensorProto.getInt32Data(i) & 0xFFFF;
                data[i] = Float.intBitsToFloat(bfloat16Bits << 16);
            }
            return Nd4j.createFromArray(data).reshape(shape);
        } else if (tensorProto.getRawData() != null && !tensorProto.getRawData().isEmpty()) {
            return createFromRawData(tensorProto.getRawData(), shape, dataType);
        } else {
            return Nd4j.zeros(dataType, shape);
        }
    }

    private static INDArray createFromRawData(ByteString rawData, long[] shape, DataType dataType) {
        ByteBuffer buffer = rawData.asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
        long totalElements = Arrays.stream(shape).reduce(1, (a, b) -> a * b);

        switch (dataType) {
            case FLOAT:
                float[] floatData = new float[(int) totalElements];
                for (int i = 0; i < totalElements; i++) {
                    floatData[i] = buffer.getFloat();
                }
                return Nd4j.createFromArray(floatData).reshape(shape);

            case DOUBLE:
                double[] doubleData = new double[(int) totalElements];
                for (int i = 0; i < totalElements; i++) {
                    doubleData[i] = buffer.getDouble();
                }
                return Nd4j.createFromArray(doubleData).reshape(shape);

            case INT:  // Deprecated, but handle it
                int[] intData = new int[(int) totalElements];
                for (int i = 0; i < totalElements; i++) {
                    intData[i] = buffer.getInt();
                }
                return Nd4j.createFromArray(intData).reshape(shape);

            case LONG: // Deprecated, but handle it
                long[] longData = new long[(int) totalElements];
                for (int i = 0; i < totalElements; i++) {
                    longData[i] = buffer.getLong();
                }
                return Nd4j.createFromArray(longData).reshape(shape);

            case SHORT: // Deprecated, but handle it
                short[] shortData = new short[(int) totalElements];
                for (int i = 0; i < totalElements; i++) {
                    shortData[i] = buffer.getShort();
                }
                return Nd4j.createFromArray(shortData).reshape(shape);

            case BYTE: // Deprecated, but handle it
            case UBYTE: // Deprecated, but handle it
                byte[] byteData = new byte[(int) totalElements];
                buffer.get(byteData);
                return Nd4j.createFromArray(byteData).reshape(shape);

            case BOOL:
                boolean[] boolData = new boolean[(int) totalElements];
                for (int i = 0; i < totalElements; i++) {
                    boolData[i] = buffer.get() != 0;
                }
                return Nd4j.createFromArray(boolData).reshape(shape);

            case HALF: // Deprecated, but handle it
                float[] halfFloatData = new float[(int) totalElements];
                for (int i = 0; i < totalElements; i++) {
                    int halfBits = buffer.getShort() & 0xFFFF;
                    halfFloatData[i] = Float.intBitsToFloat(halfToFloat(halfBits));
                }
                return Nd4j.createFromArray(halfFloatData).reshape(shape);

            case BFLOAT16:
                float[] bfloatData = new float[(int) totalElements];
                for (int i = 0; i < totalElements; i++) {
                    int bfloatBits = buffer.getShort() & 0xFFFF;
                    bfloatData[i] = Float.intBitsToFloat(bfloatBits << 16);
                }
                return Nd4j.createFromArray(bfloatData).reshape(shape);

            case UINT16:
                int[] uint16Data = new int[(int) totalElements];
                for (int i = 0; i < totalElements; i++) {
                    uint16Data[i] = buffer.getShort() & 0xFFFF;
                }
                return Nd4j.createFromArray(uint16Data).reshape(shape);

            case UINT32:
                long[] uint32Data = new long[(int) totalElements];
                for (int i = 0; i < totalElements; i++) {
                    uint32Data[i] = buffer.getInt() & 0xFFFFFFFFL;
                }
                return Nd4j.createFromArray(uint32Data).reshape(shape);

            case UINT64:
                long[] uint64Data = new long[(int) totalElements];
                for (int i = 0; i < totalElements; i++) {
                    uint64Data[i] = buffer.getLong(); // Note: Java doesn't have unsigned long
                }
                return Nd4j.createFromArray(uint64Data).reshape(shape);

            default:
                throw new IllegalArgumentException("Unsupported data type for raw data: " + dataType);
        }
    }

    // Helper methods for adding data to ONNX builders

    private static void addFloatData(Onnx.TensorProto.Builder builder, INDArray array) {
        INDArray flattened = array.reshape(-1);
        for (long i = 0; i < flattened.length(); i++) {
            builder.addFloatData(flattened.getFloat(i));
        }
    }

    private static void addDoubleData(Onnx.TensorProto.Builder builder, INDArray array) {
        INDArray flattened = array.reshape(-1);
        for (long i = 0; i < flattened.length(); i++) {
            builder.addDoubleData(flattened.getDouble(i));
        }
    }

    private static void addInt32Data(Onnx.TensorProto.Builder builder, INDArray array) {
        INDArray flattened = array.reshape(-1);
        for (long i = 0; i < flattened.length(); i++) {
            builder.addInt32Data(flattened.getInt((int) i));
        }
    }

    private static void addInt64Data(Onnx.TensorProto.Builder builder, INDArray array) {
        INDArray flattened = array.reshape(-1);
        for (long i = 0; i < flattened.length(); i++) {
            builder.addInt64Data(flattened.getLong(i));
        }
    }

    private static void addInt16Data(Onnx.TensorProto.Builder builder, INDArray array) {
        INDArray flattened = array.reshape(-1);
        for (long i = 0; i < flattened.length(); i++) {
            builder.addInt32Data(flattened.getInt((int) i)); // ONNX uses int32Data for int16
        }
    }

    private static void addByteData(Onnx.TensorProto.Builder builder, INDArray array) {
        INDArray flattened = array.reshape(-1);
        ByteBuffer buffer = ByteBuffer.allocate((int) flattened.length()).order(ByteOrder.LITTLE_ENDIAN);
        for (long i = 0; i < flattened.length(); i++) {
            buffer.put((byte) flattened.getInt((int) i));
        }
        builder.setRawData(ByteString.copyFrom(buffer.array()));
    }

    private static void addBoolData(Onnx.TensorProto.Builder builder, INDArray array) {
        INDArray flattened = array.reshape(-1);
        for (long i = 0; i < flattened.length(); i++) {
            builder.addInt32Data(flattened.getInt((int) i) != 0 ? 1 : 0); // ONNX uses int32Data for bool
        }
    }

    private static void addFloat16Data(Onnx.TensorProto.Builder builder, INDArray array) {
        INDArray flattened = array.reshape(-1);
        for (long i = 0; i < flattened.length(); i++) {
            float val = flattened.getFloat(i);
            int halfBits = floatToHalf(Float.floatToIntBits(val));
            builder.addInt32Data(halfBits); // ONNX uses int32Data for float16
        }
    }

    private static void addBFloat16Data(Onnx.TensorProto.Builder builder, INDArray array) {
        INDArray flattened = array.reshape(-1);
        for (long i = 0; i < flattened.length(); i++) {
            float val = flattened.getFloat(i);
            int bfloat16Bits = Float.floatToIntBits(val) >>> 16; // BFloat16 is top 16 bits of float32
            builder.addInt32Data(bfloat16Bits); // ONNX uses int32Data for bfloat16
        }
    }

    private static void addUInt16Data(Onnx.TensorProto.Builder builder, INDArray array) {
        INDArray flattened = array.reshape(-1);
        for (long i = 0; i < flattened.length(); i++) {
            builder.addInt32Data(flattened.getInt((int) i)); // ONNX uses int32Data for uint16
        }
    }

    private static void addUInt32Data(Onnx.TensorProto.Builder builder, INDArray array) {
        INDArray flattened = array.reshape(-1);
        for (long i = 0; i < flattened.length(); i++) {
            builder.addUint64Data(flattened.getLong(i)); // ONNX uses uint64Data for uint32
        }
    }

    private static void addUInt64Data(Onnx.TensorProto.Builder builder, INDArray array) {
        INDArray flattened = array.reshape(-1);
        for (long i = 0; i < flattened.length(); i++) {
            builder.addUint64Data(flattened.getLong(i)); // ONNX uses uint64Data for uint64
        }
    }

    // Data type conversion methods

    private static DataType onnxDataTypeToNd4j(Onnx.TensorProto.DataType onnxType) {
        switch (onnxType) {
            case FLOAT:
                return DataType.FLOAT;
            case DOUBLE:
                return DataType.DOUBLE;
            case INT32:
                return DataType.INT; // Use deprecated INT which maps to INT32
            case INT64:
                return DataType.LONG; // Use deprecated LONG which maps to INT64
            case UINT32:
                return DataType.UINT32;
            case UINT64:
                return DataType.UINT64;
            case INT16:
                return DataType.SHORT; // Use deprecated SHORT which maps to INT16
            case UINT16:
                return DataType.UINT16;
            case INT8:
                return DataType.BYTE; // Use deprecated BYTE which maps to INT8
            case UINT8:
                return DataType.UBYTE; // Use deprecated UBYTE which maps to UINT8
            case BOOL:
                return DataType.BOOL;
            case FLOAT16:
                return DataType.HALF; // Use deprecated HALF which maps to FLOAT16
            case BFLOAT16:
                return DataType.BFLOAT16;
            case STRING:
                return DataType.UTF8;
            default:
                throw new IllegalArgumentException("Unsupported ONNX data type: " + onnxType);
        }
    }

    private static Onnx.TensorProto.DataType nd4jDataTypeToOnnx(DataType nd4jType) {
        switch (nd4jType) {
            case FLOAT:
                return Onnx.TensorProto.DataType.FLOAT;
            case DOUBLE:
                return Onnx.TensorProto.DataType.DOUBLE;
            case INT: // Deprecated, but equivalent to INT32
                return Onnx.TensorProto.DataType.INT32;
            case LONG: // Deprecated, but equivalent to INT64
                return Onnx.TensorProto.DataType.INT64;
            case SHORT: // Deprecated, but equivalent to INT16
                return Onnx.TensorProto.DataType.INT16;
            case BYTE: // Deprecated, but equivalent to INT8
                return Onnx.TensorProto.DataType.INT8;
            case UBYTE: // Deprecated, but equivalent to UINT8
                return Onnx.TensorProto.DataType.UINT8;
            case BOOL:
                return Onnx.TensorProto.DataType.BOOL;
            case HALF: // Deprecated, but equivalent to FLOAT16
                return Onnx.TensorProto.DataType.FLOAT16;
            case BFLOAT16:
                return Onnx.TensorProto.DataType.BFLOAT16;
            case UINT16:
                return Onnx.TensorProto.DataType.UINT16;
            case UINT32:
                return Onnx.TensorProto.DataType.UINT32;
            case UINT64:
                return Onnx.TensorProto.DataType.UINT64;
            case UTF8:
                return Onnx.TensorProto.DataType.STRING;
            default:
                throw new IllegalArgumentException("Unsupported ND4J data type: " + nd4jType);
        }
    }

    // Half precision conversion utilities

    private static int halfToFloat(int half) {
        int sign = (half >>> 15) & 0x1;
        int exponent = (half >>> 10) & 0x1F;
        int mantissa = half & 0x3FF;

        if (exponent == 0) {
            if (mantissa == 0) {
                // Zero
                return sign << 31;
            } else {
                // Denormalized number
                exponent = 1;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x3FF;
                exponent += (127 - 15);
            }
        } else if (exponent == 0x1F) {
            if (mantissa == 0) {
                // Infinity
                return (sign << 31) | 0x7F800000;
            } else {
                // NaN
                return (sign << 31) | 0x7F800000 | (mantissa << 13);
            }
        } else {
            // Normalized number
            exponent += (127 - 15);
        }

        return (sign << 31) | (exponent << 23) | (mantissa << 13);
    }

    private static int floatToHalf(int floatBits) {
        int sign = (floatBits >>> 31) & 0x1;
        int exponent = (floatBits >>> 23) & 0xFF;
        int mantissa = floatBits & 0x7FFFFF;

        if (exponent == 0) {
            // Zero or denormalized
            return sign << 15;
        } else if (exponent == 0xFF) {
            if (mantissa == 0) {
                // Infinity
                return (sign << 15) | 0x7C00;
            } else {
                // NaN
                return (sign << 15) | 0x7C00 | (mantissa >>> 13);
            }
        } else {
            // Normalized number
            exponent -= 127 - 15;
            if (exponent <= 0) {
                // Underflow to zero or denormalized
                if (exponent < -10) {
                    return sign << 15;
                }
                mantissa = (mantissa | 0x800000) >>> (1 - exponent);
                return (sign << 15) | (mantissa >>> 13);
            } else if (exponent >= 0x1F) {
                // Overflow to infinity
                return (sign << 15) | 0x7C00;
            } else {
                return (sign << 15) | (exponent << 10) | (mantissa >>> 13);
            }
        }
    }
}