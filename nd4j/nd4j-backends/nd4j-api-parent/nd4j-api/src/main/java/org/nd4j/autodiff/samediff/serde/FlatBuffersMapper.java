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

package org.nd4j.autodiff.samediff.serde;

import lombok.SneakyThrows;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.loss.BaseLoss;
import org.nd4j.linalg.api.ops.impl.loss.bp.BaseLossBp;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.exception.ND4JUnknownDataTypeException;
import org.nd4j.shade.guava.primitives.Ints;
import com.google.flatbuffers.FlatBufferBuilder;

import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.graph.DType;
import org.nd4j.graph.FlatArray;
import org.nd4j.graph.FlatNode;
import org.nd4j.graph.FlatProperties;
import org.nd4j.graph.IntPair;
import org.nd4j.graph.OpType;
import org.nd4j.graph.VarType;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.Op.Type;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Enter;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Exit;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Merge;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.NextIteration;
import org.nd4j.linalg.api.ops.impl.controlflow.compat.Switch;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.exception.ND4UnresolvedOutputVariables;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.util.ArrayUtil;

@Slf4j
public class FlatBuffersMapper {

    private FlatBuffersMapper() {
    }


    /**
     * Convert the input byte to the equivalent
     * {@link LossReduce}, will throw an {@link IllegalArgumentException}
     * if the value is not found
     * @param input the special input
     * @return the equivalent {@link LossReduce} value if one is found
     */
    public static LossReduce getLossReduceFromByte(byte input) {
        if(input == org.nd4j.graph.LossReduce.SUM) {
            return LossReduce.SUM;
        } else if(input == org.nd4j.graph.LossReduce.NONE) {
            return LossReduce.NONE;
        } else if(input == org.nd4j.graph.LossReduce.MEAN_BY_WEIGHT) {
            return LossReduce.MEAN_BY_WEIGHT;
        } else if(input == org.nd4j.graph.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT) {
            return LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;
        } else {
            throw new IllegalArgumentException("Illegal byte did not match any known LossReduce value " + input);
        }
    }

    /**
     * Convert the {@link LossReduce}
     * enum to its flatbuffers equivalent bytes.
     * @param lossReduce the loss reduce input
     * @return
     */
    public static byte getLossFunctionAsByte(@NonNull LossReduce lossReduce) {
        switch(lossReduce) {
            case SUM:
                return org.nd4j.graph.LossReduce.SUM;
            case NONE:
                return org.nd4j.graph.LossReduce.NONE;
            case MEAN_BY_WEIGHT:
                return org.nd4j.graph.LossReduce.MEAN_BY_WEIGHT;
            case MEAN_BY_NONZERO_WEIGHT_COUNT:
                return org.nd4j.graph.LossReduce.MEAN_BY_NONZERO_WEIGHT_COUNT;
            default:
                throw new IllegalArgumentException("Illegal loss reduce " + lossReduce);

        }

    }

    /**
     * This method converts enums for DataType
     */
    public static byte getDataTypeAsByte(@NonNull DataType type) {
        switch (type) {
            case FLOAT:
                return DType.FLOAT;
            case DOUBLE:
                return DType.DOUBLE;
            case HALF:
                return DType.HALF;
            case INT:
                return DType.INT32;
            case LONG:
                return DType.INT64;
            case BOOL:
                return DType.BOOL;
            case SHORT:
                return DType.INT16;
            case BYTE:
                return DType.INT8;
            case UBYTE:
                return DType.UINT8;
            case UTF8:
                return DType.UTF8;
            case UINT16:
                return DType.UINT16;
            case UINT32:
                return DType.UINT32;
            case UINT64:
                return DType.UINT64;
            case BFLOAT16:
                return DType.BFLOAT16;
            default:
                throw new ND4JIllegalStateException("Unknown or unsupported DataType used: [" + type + "]");
        }
    }

    /**
     * This method converts enums for DataType
     */
    public static DataType getDataTypeFromByte(byte val) {
        if (val == DType.FLOAT) {
            return DataType.FLOAT;
        } else if (val == DType.DOUBLE) {
            return DataType.DOUBLE;
        } else if (val == DType.HALF) {
            return DataType.HALF;
        } else if (val == DType.INT32) {
            return DataType.INT;
        } else if (val == DType.INT64) {
            return DataType.LONG;
        } else if (val == DType.INT8) {
            return DataType.BYTE;
        } else if (val == DType.BOOL) {
            return DataType.BOOL;
        } else if (val == DType.UINT8) {
            return DataType.UBYTE;
        } else if (val == DType.INT16) {
            return DataType.SHORT;
        } else if (val == DType.UTF8) {
            return DataType.UTF8;
        } else if (val == DType.UINT16) {
            return DataType.UINT16;
        } else if (val == DType.UINT32) {
            return DataType.UINT32;
        } else if (val == DType.UINT64) {
            return DataType.UINT64;
        } else if (val == DType.BFLOAT16){
            return DataType.BFLOAT16;
        } else {
            throw new RuntimeException("Unknown datatype: " + val);
        }
    }


    /**
     * This method return operation ID for given op name/type pair.
     */
    public static long getOpNum(String name, Type type) {
        if (type == Type.LOOP) {
            return 0;
        } else if (type == Type.RETURN) {
            return 40;
        } else if (type == Type.CONDITIONAL) {
            return 10;
        } else if (type == Type.LOOP_COND) {
            return 70L;
        } else if (type == Type.LOGIC) {
            switch (name) {
                case Enter.OP_NAME:
                    return Enter.OP_NUM;
                case Exit.OP_NAME:
                    return Exit.OP_NUM;
                case NextIteration.OP_NAME:
                    return NextIteration.OP_NUM;
                case Merge.OP_NAME:
                    return Merge.OP_NUM;
                case Switch.OP_NAME:
                    return Switch.OP_NUM;
                case ExternalErrorsFunction.OP_NAME:
                    return 0;
                default:
                    throw new IllegalStateException("Unknown LOGIC op with name: " + name);
            }
        } else if (type == Type.CUSTOM) {
            val name2 = Nd4j.getExecutioner().getCustomOperations().get(name.toLowerCase());
            if (name2 == null) {
                val name3 = Nd4j.getExecutioner().getCustomOperations().get(name);
                if (name3 == null) {
                    return 0;
                } else {
                    return name3.getHash();
                }
            } else {
                return name2.getHash();
            }

        } else if(type == Type.UDF) {
            return -1;
        }  else {
            try {
                DifferentialFunction op = DifferentialFunctionClassHolder.getInstance().getInstance(name);
                return op.opNum();
            } catch (Exception e) {
                throw new RuntimeException("Could not find op number for operation: [" + name + "]", e);
            }
        }
    }


    /**
     * This method converts enums for Op.Type
     *
     * @param type Byte representing the op type
     * @return Op type
     */
    public static Type getTypeFromByte(byte type) {
        switch (type) {
            case OpType.SCALAR:
                return Type.SCALAR;
            case OpType.SCALAR_BOOL:
                return Type.SCALAR_BOOL;
            case OpType.BROADCAST:
                return Type.BROADCAST;
            case OpType.BROADCAST_BOOL:
                return Type.BROADCAST_BOOL;
            case OpType.TRANSFORM_BOOL:
                return Type.TRANSFORM_BOOL;
            case OpType.TRANSFORM_FLOAT:
                return Type.TRANSFORM_FLOAT;
            case OpType.TRANSFORM_SAME:
                return Type.TRANSFORM_SAME;
            case OpType.TRANSFORM_ANY:
                return Type.TRANSFORM_ANY;
            case OpType.TRANSFORM_STRICT:
                return Type.TRANSFORM_STRICT;
            case OpType.REDUCE_BOOL:
                return Type.REDUCE_BOOL;
            case OpType.REDUCE_LONG:
                return Type.REDUCE_LONG;
            case OpType.REDUCE_FLOAT:
                return Type.REDUCE_FLOAT;
            case OpType.REDUCE_SAME:
                return Type.REDUCE_SAME;
            case OpType.REDUCE_3:
                return Type.REDUCE3;
            case OpType.INDEX_REDUCE:
                return Type.INDEXREDUCE;
            case OpType.RANDOM:
                return Type.RANDOM;
            case OpType.LOGIC:
                return Type.LOGIC;
            case OpType.CUSTOM:
                return Type.CUSTOM;
            case OpType.PAIRWISE:
                return Type.PAIRWISE;
            case OpType.PAIRWISE_BOOL:
                return Type.PAIRWISE_BOOL;
            case OpType.SUMMARYSTATS:
                return Type.SUMMARYSTATS;
            case OpType.UDF:
                return Type.UDF;
            default:
                throw new UnsupportedOperationException("Unknown op type passed in: " + type);
        }
    }

    /**
     * This method converts an Op.Type to it's corresponding byte value
     *
     * @param type type to convert
     * @return Byte representing the op type
     */
    public static byte getFlatOpType(Type type) {
        switch (type) {
            case SCALAR:
                return OpType.SCALAR;
            case SCALAR_BOOL:
                return OpType.SCALAR_BOOL;
            case BROADCAST:
                return OpType.BROADCAST;
            case BROADCAST_BOOL:
                return OpType.BROADCAST_BOOL;
            case TRANSFORM_BOOL:
                return OpType.TRANSFORM_BOOL;
            case TRANSFORM_FLOAT:
                return OpType.TRANSFORM_FLOAT;
            case TRANSFORM_SAME:
                return OpType.TRANSFORM_SAME;
            case TRANSFORM_ANY:
                return OpType.TRANSFORM_ANY;
            case TRANSFORM_STRICT:
                return OpType.TRANSFORM_STRICT;
            case SPECIAL:
                return OpType.TRANSFORM_STRICT;
            case REDUCE_FLOAT:
                return OpType.REDUCE_FLOAT;
            case REDUCE_BOOL:
                return OpType.REDUCE_BOOL;
            case REDUCE_SAME:
                return OpType.REDUCE_SAME;
            case REDUCE_LONG:
                return OpType.REDUCE_LONG;
            case REDUCE3:
                return OpType.REDUCE_3;
            case INDEXREDUCE:
                return OpType.INDEX_REDUCE;
            case RANDOM:
                return OpType.RANDOM;
            case CONDITIONAL:
            case LOOP:
            case RETURN:
            case LOOP_COND:
            case LOGIC:
                return OpType.LOGIC;
            case CUSTOM:
                return OpType.CUSTOM;
            case PAIRWISE:
                return OpType.PAIRWISE;
            case PAIRWISE_BOOL:
                return OpType.PAIRWISE_BOOL;
            case SUMMARYSTATS:
            case VARIANCE:
                return OpType.SUMMARYSTATS;
            case UDF:
                return OpType.UDF;
            default:
                throw new UnsupportedOperationException("Unknown op type passed in: " + type);
        }
    }


    /**
     * This method just converts enums
     */
    public static ByteOrder getOrderFromByte(byte val) {
        if (val == org.nd4j.graph.ByteOrder.LE) {
            return ByteOrder.LITTLE_ENDIAN;
        } else {
            return ByteOrder.BIG_ENDIAN;
        }
    }

    /**
     * This method returns current byte order for this JVM as libnd4j enum
     */
    public static byte getOrderAsByte() {
        if (ByteOrder.nativeOrder().equals(ByteOrder.BIG_ENDIAN)) {
            return org.nd4j.graph.ByteOrder.BE;
        } else {
            return org.nd4j.graph.ByteOrder.LE;
        }
    }

    public static DifferentialFunction fromFlatNode(FlatNode fn) {

        int id = fn.id();               //ID of the node
        String name = fn.name();        //Name of the node, NOT the name of the op
        Type opType = FlatBuffersMapper.getTypeFromByte(fn.opType());
        long opNum = fn.opNum();        //Op num: hash for custom, number for legacy
        int[] input = new int[fn.inputLength()];
        for (int i = 0; i < input.length; i++) {
            input[i] = fn.input(i);
        }
        IntPair[] inputPaired = new IntPair[fn.inputPairedLength()];
        for (int i = 0; i < inputPaired.length; i++) {
            inputPaired[i] = fn.inputPaired(i);
        }
        int[] output = new int[fn.outputLength()];
        for (int i = 0; i < output.length; i++) {
            output[i] = fn.output(i);
        }
        double[] extraParams = new double[fn.extraParamsLength()];
        for (int i = 0; i < extraParams.length; i++) {
            extraParams[i] = fn.extraParams(i);
        }
        long[] extraInteger = new long[fn.extraIntegerLength()];
        for (int i = 0; i < extraInteger.length; i++) {
            extraInteger[i] = fn.extraInteger(i);
        }
        boolean[] extraBools = new boolean[fn.extraBoolsLength()];
        for (int i = 0; i < extraBools.length; i++) {
            extraBools[i] = fn.extraBools(i);
        }
        DataType[] extraDTypes = new DataType[fn.extraTypesLength()];
        for (int i = 0; i < extraDTypes.length; i++) {
            extraDTypes[i] = DataType.fromInt(fn.extraTypes(i));
        }

        String[] extraStrings = new String[fn.extraStringsLength()];
        for (int i = 0; i < extraStrings.length; i++) {
            extraStrings[i] = fn.extraStrings(i);
        }

        long[] dimensions = new long[fn.dimensionsLength()];
        for (int i = 0; i < dimensions.length; i++) {
            dimensions[i] = fn.dimensions(i);
        }
        FlatArray fa = fn.scalar();
        INDArray scalar = null;

        // Handle scalar arrays with special care
        if (fa != null) {
            // Check if this is a scalar (rank 0) array
            if (fa.shapeLength() == 0) {
                // Log the scalar case for debugging
                log.debug("Processing scalar array in FlatNode. FlatArray dtype: {}", fa.dtype());

                // For scalar arrays, we need to ensure the extras value contains the proper data type
                byte dtype = fa.dtype();
                if (dtype == 0) {
                    log.warn("FlatArray has 0 dtype for scalar. Defaulting to FLOAT.");
                    dtype = FlatBuffersMapper.getDataTypeAsByte(DataType.FLOAT);
                }

                DataType dataType = FlatBuffersMapper.getDataTypeFromByte(dtype);

                // Create scalar array manually with proper extras containing data type
                scalar = Nd4j.scalar(dataType, 0.0);

                // Read actual value from buffer if available
                ByteBuffer bb = fa.bufferAsByteBuffer();
                if (bb != null && bb.remaining() > 0) {
                    bb.position(0); // Reset position
                    switch (dataType) {
                        case FLOAT:
                            scalar.putScalar(0, bb.getFloat());
                            break;
                        case DOUBLE:
                            scalar.putScalar(0, bb.getDouble());
                            break;
                        case INT:
                            scalar.putScalar(0, bb.getInt());
                            break;
                        // Add cases for other data types as needed
                        default:
                            log.warn("Unhandled scalar data type: {}. Using zero value.", dataType);
                    }
                }
            } else {
                // Non-scalar case, use standard method
                scalar = Nd4j.createFromFlatArray(fa);
            }
        }

        FlatProperties[] flatProperties = new FlatProperties[fn.propertiesLength()];
        for (int i = 0; i < flatProperties.length; i++) {
            flatProperties[i] = fn.properties(i);
        }
        Map<String, Object> props = FlatBuffersMapper
                .mapFlatPropertiesToFunctionProperties(Arrays.asList(flatProperties));

        if (opType == Type.CUSTOM || opType == Type.LOGIC || opType == Type.UDF) {
            String opName = fn.opName();

            DifferentialFunction op;
            Class<?> c = DifferentialFunctionClassHolder.getInstance().customOpClassForHashAndName(opNum, opName);

            Preconditions.checkNotNull(c, "Could not find class for hash %s", opNum);

            try {
                op = (DifferentialFunction) c.newInstance();
            } catch (IllegalAccessException | InstantiationException e) {
                throw new RuntimeException("Error creating differential function instance of type " + c);
            }

            op.setOwnName(name);

            //Set input SDVariables:

            //Set args:
            if(op instanceof CustomOp) {
                ((CustomOp) op).addIArgument(extraInteger);
                ((CustomOp) op).addTArgument(extraParams);
                ((CustomOp) op).addBArgument(extraBools);
                ((CustomOp) op).addDArgument(extraDTypes);
                ((CustomOp) op).addSArgument(extraStrings);
            }

            //base loss gets saved as an int argument, ensure that the field is set
            if(op instanceof BaseLoss && extraInteger != null && extraInteger.length > 0) {
                BaseLoss baseLoss = (BaseLoss) op;
                baseLoss.setLossReduce(LossReduce.values()[(int) extraInteger[0]]);
            } else if(op instanceof BaseLossBp && extraInteger != null && extraInteger.length > 0) {
                BaseLossBp baseLossBp = (BaseLossBp) op;
                baseLossBp.setLossReduce(LossReduce.values()[(int) extraInteger[0]]);
            }

            op.setPropertiesForFunction(props);
            if(op instanceof CustomOp)
                ((CustomOp) op).configureFromArguments();
            return op;
        } else {
            Class<?> c = LegacyOpMapper.getLegacyOpClassForId(opType, (int) opNum);
            Op op;
            try {
                op = (Op) c.newInstance();
            } catch (IllegalAccessException | InstantiationException e) {
                throw new RuntimeException("Error creating differential function (Op) instance of type " + c);
            }

            if (extraParams.length > 0) {
                //Assume that extraParams length 0 means extraArgs was originally null, NOT originally length 0
                Object[] extraParamsObj = new Object[extraParams.length];
                for (int i = 0; i < extraParams.length; i++) {
                    extraParamsObj[i] = extraParams[i];
                }
                op.setExtraArgs(extraParamsObj);
            }
            if (opType == Type.SCALAR || opType == Type.SCALAR_BOOL) {
                ScalarOp sOp = (ScalarOp) op;
                sOp.setScalar(scalar);
            } else if (opType == Type.REDUCE_FLOAT || opType == Type.REDUCE3 || opType == Type.SUMMARYSTATS
                    || opType == Type.VARIANCE
                    || opType == Type.REDUCE_BOOL || opType == Type.REDUCE_LONG
                    || opType == Type.REDUCE_SAME) {
                val ba = (BaseReduceOp) op; //Reduce3 ops are also all BaseAccumulations
                ba.setDimensions(dimensions);
                ba.setDimensionz(Shape.ndArrayDimFromLong(dimensions));
                if(extraBools.length > 0)
                    ba.setKeepDims(extraBools[0]);

            } else if (opType == Type.INDEXREDUCE) {
                BaseIndexAccumulation bia = (BaseIndexAccumulation) op;
                bia.setDimensions(dimensions);
                bia.setDimensionz(Shape.ndArrayDimFromLong(dimensions));
                if(extraBools.length > 0)
                    bia.setKeepDims(extraBools[0]);
            }
        /*
        Op types that don't need any extra/special mapping:
        TRANSFORM_BOOL - BooleanNot, IsFinite, IsInf, IsNaN, MatchConditionTransform
        TRANSFORM_ANY - IsMax, Assign
        TRANSFORM_FLOAT - Histogram, Sqrt
        TRANSFORM_STRICT - Cos, Log, Sigmoid, etc
        TRANSFORM_SAME - Abs, Ceil, etc
         */

            ((DifferentialFunction) op).setPropertiesForFunction(props);
            return (DifferentialFunction) op;
        }
    }

    private static final boolean[] EMPTY_BOOLEAN = new boolean[0];
    private static final int[] EMPTY_INT = new int[0];
    private static final long[] EMPTY_LONG = new long[0];
    private static final double[] EMPTY_DOUBLE = new double[0];

    private static final byte[] EMPTY_BYTE = new byte[0];

    public static Map<String, Object> mapFlatPropertiesToFunctionProperties(Iterable<FlatProperties> list) {
        Map<String, Object> out = new HashMap<>();
        for (FlatProperties p : list) {

            String name = p.name();
            //Work out type:
            if (p.shapeLength() > 0) {
                //Array type
                int[] shape = new int[p.shapeLength()];
                for (int i = 0; i < shape.length; i++) {
                    shape[i] = p.shape(i);
                }

                if (p.iLength() > 0) {
                    int[] iArr = new int[p.iLength()];
                    for (int i = 0; i < iArr.length; i++) {
                        iArr[i] = p.i(i);
                    }
                    if (shape.length == 0 || shape.length == 1) {
                        out.put(name, iArr);
                    } else if (shape.length == 2) {
                        out.put(name, ArrayUtil.reshapeInt(iArr, shape[0], shape[1]));
                    } else if (shape.length == 3) {
                        out.put(name, ArrayUtil.reshapeInt(iArr, shape[0], shape[1], shape[2]));
                    }
                } else if (p.dLength() > 0) {
                    double[] dArr = new double[p.dLength()];
                    for (int i = 0; i < dArr.length; i++) {
                        dArr[i] = p.d(i);
                    }
                    if (shape.length == 0 || shape.length == 1) {
                        out.put(name, dArr);
                    } else if (shape.length == 2) {
                        out.put(name, ArrayUtil.reshapeDouble(dArr, shape[0], shape[1]));
                    } else if (shape.length == 3) {
                        out.put(name, ArrayUtil.reshapeDouble(dArr, shape[0], shape[1], shape[2]));
                    }
                } else if (p.lLength() > 0) {
                    long[] lArr = new long[p.lLength()];
                    for (int i = 0; i < lArr.length; i++) {
                        lArr[i] = p.l(i);
                    }
                    if (shape.length == 0 || shape.length == 1) {
                        out.put(name, lArr);
                    } else if (shape.length == 2) {
                        out.put(name, ArrayUtil.reshapeLong(lArr, shape[0], shape[1]));
                    } else if (shape.length == 3) {
                        out.put(name, ArrayUtil.reshapeLong(lArr, shape[0], shape[1], shape[2]));
                    }
                } else if (p.bLength() > 0) {
                    boolean[] bArr = new boolean[p.bLength()];
                    for (int i = 0; i < bArr.length; i++) {
                        bArr[i] = p.b(i);
                    }
                    if (shape.length == 0 || shape.length == 1) {
                        out.put(name, bArr);
                    } else if (shape.length == 2) {
                        out.put(name, ArrayUtil.reshapeBoolean(bArr, shape[0], shape[1]));
                    } else if (shape.length == 3) {
                        out.put(name, ArrayUtil.reshapeBoolean(bArr, shape[0], shape[1], shape[2]));
                    }
                } else if (p.sLength() > 0) {
                    String[] sArr = new String[p.sLength()];
                    for (int i = 0; i < sArr.length; i++) {
                        sArr[i] = p.s(i);
                    }
                    if (shape.length == 0 || shape.length == 1) {
                        out.put(name, sArr);
                    } else if (shape.length == 2) {
                        out.put(name, ArrayUtil.reshapeObject(sArr, shape[0], shape[1]));
                    } else if (shape.length == 3) {
                        out.put(name, ArrayUtil.reshapeObject(sArr, shape[0], shape[1], shape[2]));
                    }
                } else if (p.aLength() > 0) {
                    INDArray[] iArr = new INDArray[p.aLength()];
                    for (int i = 0; i < iArr.length; i++) {
                        FlatArray fa = p.a(0);
                        iArr[i] = Nd4j.createFromFlatArray(fa);
                    }
                    if (shape.length == 0 || shape.length == 1) {
                        out.put(name, iArr);
                    } else if (shape.length == 2) {
                        out.put(name, ArrayUtil.reshapeObject(iArr, shape[0], shape[1]));
                    } else if (shape.length == 3) {
                        out.put(name, ArrayUtil.reshapeObject(iArr, shape[0], shape[1], shape[2]));
                    }
                } else {
                    //null property case
                    out.put(name, null);
                }
            } else {
                //non-array primitive, String or INDArray
                if (p.bLength() > 0) {
                    out.put(name, p.b(0));
                } else if (p.iLength() > 0) {
                    out.put(name, p.i(0));
                } else if (p.lLength() > 0) {
                    out.put(name, p.l(0));
                } else if (p.dLength() > 0) {
                    out.put(name, p.d(0));
                } else if (p.sLength() > 1) {
                    // FIXED: Multiple strings - return as array
                    String[] sArr = new String[p.sLength()];
                    for (int i = 0; i < sArr.length; i++) {
                        sArr[i] = p.s(i);
                    }
                    out.put(name, sArr);
                } else if (p.sLength() == 1) {
                    // FIXED: Single string - return as string
                    out.put(name, p.s(0));
                } else if (p.aLength() > 0) {
                    FlatArray fa = p.a(0);
                    out.put(name, Nd4j.createFromFlatArray(fa));
                } else {
                    //null property case
                    out.put(name, null);
                }
            }
        }
        return out;
    }

    /**
     * Converts a SameDiff DifferentialFunction node into its FlatBuffers representation.
     * Added isCloneContext flag to prevent serialization of INDArray properties during cloning.
     *
     * @param sameDiff        The SameDiff instance (needed for context like variable lookups).
     * @param node            The DifferentialFunction (op) to serialize.
     * @param bufferBuilder   The FlatBufferBuilder instance.
     * @param variables       List of all SDVariables in the graph (order matters for output mapping).
     * @param reverseMap      Map from Variable Name to Node ID (op ID or independent var ID).
     * @param forwardMap      Map for forward declarations (like loops).
     * @param framesMap       Map for control flow frame names to IDs.
     * @param idCounter       Counter for generating new node IDs.
     * @param id              Pre-assigned node ID (if available, otherwise generated).
     * @param isCloneContext  If true, skips serializing INDArray properties and ScalarOp arrays (used by cloneViaSerialize).
     * @return The offset of the created FlatNode in the buffer.
     * @throws IOException If serialization fails.
     */
    public static int asFlatNode(@NonNull SameDiff sameDiff, @NonNull DifferentialFunction node, @NonNull FlatBufferBuilder bufferBuilder, List<SDVariable> variables,
                                 Map<String, Integer> reverseMap, Map<String, Integer> forwardMap, Map<String, Integer> framesMap, AtomicInteger idCounter, Integer id,
                                 boolean isCloneContext) throws IOException { // Added isCloneContext flag
        val opName = node.opName();
        val opTypeEnum = node.opType();
        val hash = FlatBuffersMapper.getOpNum(opName, opTypeEnum); // Assume helper handles null type ok
        val nodeOwnName = node.getOwnName();
        if (nodeOwnName == null) throw new ND4JIllegalStateException("Node ownName cannot be null for serialization.");

        // --- Extract Op Args ---
        double[] extras = EMPTY_DOUBLE;
        boolean[] boolArgs = EMPTY_BOOLEAN;
        byte[] dtypeArgs = EMPTY_BYTE;
        long[] extraBits = EMPTY_LONG;
        int[] extraStringIds = EMPTY_INT;
        if (node instanceof CustomOp) {
            CustomOp op = (CustomOp) node;
            if(op.tArgs() != null) extras = op.tArgs(); if(op.iArgs() != null)
                extraBits = op.iArgs();
            if(op.bArgs() != null)
                boolArgs = op.bArgs();
            if (op.numDArguments() > 0) {
                DataType[] dTypes = op.dArgs();
                if(dTypes != null) {
                    dtypeArgs = new byte[dTypes.length];
                    for (int e = 0; e < dtypeArgs.length; e++)
                        dtypeArgs[e] = getDataTypeAsByte(dTypes[e]);
                }
            }
            if(op.numSArguments() > 0) {
                String[] sArgs = op.sArgs();
                if(sArgs != null) {
                    extraStringIds = new int[sArgs.length];
                    for(int i = 0; i < sArgs.length; i++)
                        extraStringIds[i] = bufferBuilder.createString(sArgs[i] != null ? sArgs[i] : "");
                }
            }
        } else {
            Object[] eArgs = node.getExtraArgs();
            if (eArgs != null) {
                extras = new double[eArgs.length];
                for (int e = 0; e < extras.length; e++) {
                    if(eArgs[e] instanceof Number)
                        extras[e] = ((Number) eArgs[e]).doubleValue();
                }
            }
        }
        if (node instanceof Enter) {
            val frameName = ((Enter) node).getFrameName();
            if (frameName != null) {
                if (!framesMap.containsKey(frameName))
                    framesMap.put(frameName, idCounter.incrementAndGet());
                extraBits = new long[]{framesMap.get(frameName).intValue()};
                extraStringIds = new int[]{bufferBuilder.createString(frameName)};
            }
        }
        if (node instanceof ReduceOp || node instanceof IndexAccumulation) {
            boolean currentKeepDims = false;
            if (node instanceof ReduceOp)
                currentKeepDims = ((ReduceOp) node).isKeepDims();
            else if (node instanceof IndexAccumulation)
                currentKeepDims = ((IndexAccumulation) node).isKeepDims();
            if (boolArgs == EMPTY_BOOLEAN)
                boolArgs = new boolean[2];
            else if (boolArgs.length < 2)
                boolArgs = Arrays.copyOf(boolArgs, 2);
            boolArgs[0] = currentKeepDims;
            boolArgs[1] = true;
        }
        // --- End Extract Op Args ---

        // --- Input Processing ---
        val inPaired = new ArrayList<Integer>();
        SDVariable[] inputs = null;
        try { inputs = node.args(); }
        catch (Exception e) {
            log.error("Failed resolve inputs for op '{}'.", nodeOwnName, e);
            inputs = new SDVariable[0];
        }
        if (inputs != null) {
            for (SDVariable input : inputs) {
                if (input == null || input.name() == null)
                    continue;
                String varName = input.name();
                int outIdx = 0;
                Variable inputMeta = sameDiff.getVariables().get(varName);
                if (inputMeta == null) { if (!reverseMap.containsKey(varName)) {
                    int fwdNodeId = idCounter.incrementAndGet(); forwardMap.put(varName, fwdNodeId);
                    reverseMap.put(varName, fwdNodeId);
                    log.warn("Temp ID {} for missing input '{}'.", fwdNodeId, varName);
                }
                } else if (inputMeta.getOutputOfOp() != null) {
                    SameDiffOp producingSdo = sameDiff.getOps().get(inputMeta.getOutputOfOp());
                    if (producingSdo != null) {
                        List<String> pOuts = producingSdo.getOutputsOfOp();
                        if (pOuts != null) { int fIdx = pOuts.indexOf(varName);
                            if (fIdx >= 0) outIdx = fIdx;
                        }
                    }
                }
                if (!reverseMap.containsKey(varName)) {
                    if (varName != null && varName.startsWith("NextIteration")) {
                        int fwdNodeId = idCounter.incrementAndGet();
                        forwardMap.put(varName, fwdNodeId); reverseMap.put(varName, fwdNodeId);
                    } else {
                        throw new ND4JIllegalStateException("Unknown input var: [" + varName + "] for op " + nodeOwnName);
                    }
                }
                int nodeId = reverseMap.get(varName);
                inPaired.add(IntPair.createIntPair(bufferBuilder, nodeId, outIdx));
            }
        }
        // --- End Input Processing ---

        // --- ID assignment & Output Name Mapping ---
        int ownId = id != null ? id : (forwardMap.containsKey(nodeOwnName) ? forwardMap.get(nodeOwnName) :
                idCounter.incrementAndGet());
        String[] outNames = node.outputVariablesNames();
        if(outNames == null) outNames = new String[0];
        for (String s : outNames) {
            if(s == null)
                continue;
            if (!reverseMap.containsKey(s))
                reverseMap.put(s, ownId);
        }
        // --- End ID assignment & Output Name Mapping ---

        // --- Dimension Processing ---
        int[] dims = EMPTY_INT; Op.Type t = node.opType();
        if (t == Type.REDUCE_FLOAT || t == Type.REDUCE_SAME ||
                t == Type.REDUCE_BOOL || t == Type.REDUCE_LONG ||
                t == Type.INDEXREDUCE || t == Type.REDUCE3 ||
                t == Type.VARIANCE ||
                t == Type.SUMMARYSTATS) {
            long[] nodeDims = node.getDimensions();
            if (nodeDims != null)
                dims = ArrayUtil.toInts(nodeDims);
        }
        // --- End Dimension Processing ---

        // --- Properties Processing ---
        Map<String, Object> fnProps = node.propertiesForFunction();
        int propIdx = 0;
        if(fnProps != null && !fnProps.isEmpty()) {
            int[] flatProps = FlatBuffersMapper.mapFunctionPropertiesToFlatProperties(bufferBuilder, fnProps, isCloneContext);
            if (flatProps != null && flatProps.length > 0) {
                propIdx = FlatNode.createPropertiesVector(bufferBuilder, flatProps);
            }
        }
        // --- End Properties Processing ---

        // --- Create FlatBuffer Vectors for Node ---
        int nodesInPaired = FlatNode.createInputPairedVector(bufferBuilder, Ints.toArray(inPaired));
        int nodesOut = FlatNode.createOutputVector(bufferBuilder, EMPTY_INT);
        int extraz = FlatNode.createExtraParamsVector(bufferBuilder, extras);
        int integerArgs = FlatNode.createExtraIntegerVector(bufferBuilder, extraBits);
        int bArgs = FlatNode.createExtraBoolsVector(bufferBuilder, boolArgs);
        int dArgsOffset = FlatNode.createExtraTypesVector(bufferBuilder, dtypeArgs);
        int dimensionsOffset = FlatNode.createDimensionsVector(bufferBuilder, dims);
        int fname = bufferBuilder.createString(nodeOwnName);
        int scopeNameOffset = bufferBuilder.createString("");
        int sArgs3 = FlatNode.createExtraStringsVector(bufferBuilder, extraStringIds);

        // --- Scalar Processing (Skip array if cloning) ---
        int scalarOffset = 0;
        if (node instanceof ScalarOp && !isCloneContext) { // Check flag
            ScalarOp sOp = (ScalarOp) node;
            INDArray s = sOp.scalar();
            if (s != null) {
                try {
                    scalarOffset = SameDiffSerializer.serializeSmallNdArrayToFlatBuffer(s, bufferBuilder); } // Use helper directly
                catch (Exception e) {
                    log.warn("Failed to serialize scalar array for op '{}'.", nodeOwnName, e); scalarOffset = 0;
                }
            }
        }
        // --- End Scalar Processing ---

        // --- Output Variable Names and Types ---
        List<String> currentOutVarNames = Collections.emptyList();
        SameDiffOp currentSDO = sameDiff.getOps().get(nodeOwnName);
        if (currentSDO != null && currentSDO.getOutputsOfOp() != null) {
            currentOutVarNames = currentSDO.getOutputsOfOp();
        }

        else if (outNames.length > 0) {
            currentOutVarNames = Arrays.asList(outNames);
        }
        int[] outVarNamesStringsOffsets = new int[currentOutVarNames.size()];
        List<Byte> outTypeList = new ArrayList<>();
        for (int i = 0; i < currentOutVarNames.size(); i++) {
            String s = currentOutVarNames.get(i);
            if (s == null) {
                outVarNamesStringsOffsets[i] = bufferBuilder.createString("");
                continue;
            }
            outVarNamesStringsOffsets[i] = bufferBuilder.createString(s);
            SDVariable v = sameDiff.getVariable(s);
            if (v != null) {
                outTypeList.add(FlatBuffersMapper.getDataTypeAsByte(v.dataType()));
            } else {
                log.warn("Output var '{}' for op '{}' not found for type lookup.", s, nodeOwnName);
            }
        }
        int outVarNamesOffset = FlatNode.createOutputNamesVector(bufferBuilder, outVarNamesStringsOffsets);
        byte[] finalOutTypes = new byte[outTypeList.size()];
        for (int k = 0; k < outTypeList.size(); k++) {
            finalOutTypes[k] = outTypeList.get(k);
        }
        int outTypesOffset = FlatNode.createOutputTypesVector(bufferBuilder, finalOutTypes);
        // --- End Output Variable Handling ---

        int opNameOffset = bufferBuilder.createString(opName != null ? opName : "");

        // --- Control dependencies ---
        int opCdsOffset = 0, varCdsOffset = 0, cdsForOffset = 0;
        if(currentSDO != null) {
            int[] opCdsArr = mapOrNull(currentSDO.getControlDeps(), bufferBuilder);
            if(opCdsArr != null) opCdsOffset = FlatNode.createControlDepsVector(bufferBuilder, opCdsArr);
            int[] varCdsArr = mapOrNull(currentSDO.getVarControlDeps(), bufferBuilder);
            if(varCdsArr != null)
                varCdsOffset = FlatNode.createVarControlDepsVector(bufferBuilder, varCdsArr);
            int[] cdsForArr = mapOrNull(currentSDO.getControlDepFor(), bufferBuilder);
            if(cdsForArr != null)
                cdsForOffset = FlatNode.createControlDepForVector(bufferBuilder, cdsForArr);
        }
        else {
            log.warn("SameDiffOp metadata not found for op '{}'. Control dependencies missing.", nodeOwnName);
        }
        // --- End Control dependencies ---

        // --- Determine Final OpType Byte ---
        byte finalOpTypeByte; if (opTypeEnum != null) {
            finalOpTypeByte = FlatBuffersMapper.getFlatOpType(opTypeEnum);
        } else {
            log.warn("Op Type null for node '{}'. Defaulting to CUSTOM.", nodeOwnName);
            finalOpTypeByte = OpType.CUSTOM;
        }
        // --- End OpType Handling ---

        // *** Final createFlatNode call ***
        FlatNode.startFlatNode(bufferBuilder);
        FlatNode.addId(bufferBuilder, ownId); FlatNode.addName(bufferBuilder, fname);
        FlatNode.addOpType(bufferBuilder, finalOpTypeByte);
        FlatNode.addOpNum(bufferBuilder, hash);
        FlatNode.addProperties(bufferBuilder, propIdx);
        FlatNode.addInputPaired(bufferBuilder, nodesInPaired);
        FlatNode.addOutput(bufferBuilder, nodesOut);
        FlatNode.addExtraParams(bufferBuilder, extraz);
        FlatNode.addExtraInteger(bufferBuilder, integerArgs);
        FlatNode.addExtraBools(bufferBuilder, bArgs);
        FlatNode.addDimensions(bufferBuilder, dimensionsOffset);
        FlatNode.addDevice(bufferBuilder, -1);
        FlatNode.addScopeId(bufferBuilder, 0);
        FlatNode.addScopeName(bufferBuilder, scopeNameOffset);
        FlatNode.addOutputNames(bufferBuilder, outVarNamesOffset);
        FlatNode.addOpName(bufferBuilder, opNameOffset);
        FlatNode.addOutputTypes(bufferBuilder, outTypesOffset);
        FlatNode.addScalar(bufferBuilder, scalarOffset);
        FlatNode.addControlDeps(bufferBuilder, opCdsOffset);
        FlatNode.addVarControlDeps(bufferBuilder, varCdsOffset);
        FlatNode.addControlDepFor(bufferBuilder, cdsForOffset);
        FlatNode.addExtraTypes(bufferBuilder, dArgsOffset);
        FlatNode.addExtraStrings(bufferBuilder, sArgs3);
        int flatNode = FlatNode.endFlatNode(bufferBuilder);

        return flatNode;
    }




    public static int asFlatNode(@NonNull SameDiff sameDiff, @NonNull DifferentialFunction node, @NonNull FlatBufferBuilder bufferBuilder, List<SDVariable> variables,
                                 Map<String, Integer> reverseMap, Map<String, Integer> forwardMap, Map<String, Integer> framesMap, AtomicInteger idCounter, Integer id) throws IOException {
        return asFlatNode(sameDiff, node, bufferBuilder, variables, reverseMap, forwardMap, framesMap, idCounter, id, false); // Default isCloneContext to false
    }
    /**
     * Maps function properties to FlatBuffer property objects, optionally skipping INDArray/SDVariable properties
     * if in clone context, and validating INDArray rank/shape before serialization otherwise.
     *
     * @param fbb            FlatBufferBuilder
     * @param fnProps        Map of properties from DifferentialFunction.propertiesForFunction()
     * @param isCloneContext If true, skip array properties.
     * @return Array of offsets to FlatProperties tables, or empty array if no valid properties found.
     */
    public static int[] mapFunctionPropertiesToFlatProperties(FlatBufferBuilder fbb, Map<String, Object> fnProps, boolean isCloneContext) {
        if (fnProps == null || fnProps.isEmpty()) {
            return EMPTY_INT;
        }
        List<Integer> outOffsets = new ArrayList<>();

        List<Map.Entry<String, Object>> sortedEntries = new ArrayList<>(fnProps.entrySet());
        sortedEntries.sort(Map.Entry.comparingByKey());

        for (Map.Entry<String, Object> e : sortedEntries) {
            Object v = e.getValue();
            String key = e.getKey();
            if (key == null) {
                log.warn("Skipping null property key.");
                continue;
            }

            // --- Skip logic for Clone Context ---
            if (isCloneContext) {
                if (v instanceof INDArray || v instanceof SDVariable ||
                        (v != null && v.getClass().isArray() && (v instanceof INDArray[] || v instanceof SDVariable[]))) {
                    log.trace("Skipping Array/Variable property '{}' during op cloning.", key);
                    continue;
                }
            }
            // --- End Skip logic ---

            int iname = fbb.createString(key);
            int[] i_arr = null;
            long[] l_arr = null;
            double[] d_arr = null;
            int[] aIdx = null;
            boolean[] b_arr = null;
            int[] sIdx = null;
            int[] shape = null;

            // --- Handle different property types ---
            if (v == null) {
                /* Defaults handle null case */
            }
            else if (v instanceof Boolean) {
                b_arr = new boolean[]{(Boolean) v};
                shape = EMPTY_INT;
            }
            else if (v instanceof Character) {
                i_arr = new int[]{(Character) v};
                shape = EMPTY_INT;
            }
            else if (v instanceof Integer) {
                i_arr = new int[]{(Integer) v};
                shape = EMPTY_INT;
            }
            else if (v instanceof Long) {
                l_arr = new long[]{(Long) v};
                shape = EMPTY_INT;
            }
            else if (v instanceof Float) {
                d_arr = new double[]{(Float) v};
                shape = EMPTY_INT;
            }
            else if (v instanceof Double) {
                d_arr = new double[]{(Double) v};
                shape = EMPTY_INT;
            }
            else if (v instanceof Number) {
                d_arr = new double[]{((Number)v).doubleValue()};
                shape = EMPTY_INT;
            }
            else if (v instanceof String) {
                sIdx = new int[]{fbb.createString((String) v)};
                shape = EMPTY_INT;
            }
            else if (v instanceof String[]) {
                String[] stringArr = (String[]) v;
                sIdx = new int[stringArr.length];
                for (int i = 0; i < stringArr.length; i++) {
                    sIdx[i] = fbb.createString(stringArr[i]);
                }
                shape = new int[]{stringArr.length};
            }
            else if (v instanceof DataType) {
                sIdx = new int[]{fbb.createString(v.toString())};
                shape = EMPTY_INT;
            }
            else if (v instanceof SDVariable) {
                sIdx = new int[]{fbb.createString(((SDVariable) v).name())};
                shape = EMPTY_INT;
            }
            else if (v instanceof SDVariable[]) {
                SDVariable[] sdVarArr = (SDVariable[]) v;
                sIdx = new int[sdVarArr.length];
                for (int i = 0; i < sdVarArr.length; i++) {
                    sIdx[i] = fbb.createString(sdVarArr[i] != null ? sdVarArr[i].name() : "");
                }
                shape = new int[]{sdVarArr.length};
            }
            else if (v instanceof Enum) {
                sIdx = new int[]{fbb.createString(v.toString())};
                shape = EMPTY_INT;
            }
            else if (v instanceof INDArray) {
                INDArray arr = (INDArray) v;

                // FATAL VALIDATION: Check shape buffer before calling toFlatArray
                try {
                    long[] shapeInfo = arr.shapeInfoDataBuffer().asLong();
                    DataType extractedType = ArrayOptionsHelper.dataType(shapeInfo);
                    if (extractedType == null || extractedType == DataType.UNKNOWN) {
                        throw new IllegalStateException(String.format(
                                "FATAL: INVALID SHAPE BUFFER in property '%s'. ArrayOptionsHelper.dataType() returned NULL/UNKNOWN. " +
                                        "ShapeInfo: %s. TERMINATING TO PREVENT CORRUPTION.",
                                e.getKey(), Arrays.toString(shapeInfo)));
                    }
                    if (extractedType != arr.dataType()) {
                        throw new IllegalStateException(String.format(
                                "FATAL: SHAPE BUFFER MISMATCH in property '%s'. Buffer says %s, array says %s. " +
                                        "ShapeInfo: %s. TERMINATING TO PREVENT CORRUPTION.",
                                e.getKey(), extractedType, arr.dataType(), Arrays.toString(shapeInfo)));
                    }
                } catch (ND4JUnknownDataTypeException ex) {
                    throw new IllegalStateException(String.format(
                            "FATAL: CORRUPT SHAPE BUFFER in property '%s'. %s. TERMINATING TO PREVENT CORRUPTION.",
                            e.getKey(), ex.getMessage()), ex);
                }

                aIdx = new int[]{arr.toFlatArray(fbb)};
            }
            else if (v.getClass().isArray()) {
                if (v.getClass().getComponentType().isPrimitive()) {
                    if (v instanceof boolean[]) {
                        b_arr = (boolean[]) v;
                        shape = new int[]{b_arr.length};
                    }
                    else if (v instanceof int[]) {
                        i_arr = (int[]) v;
                        shape = new int[]{i_arr.length};
                    }
                    else if (v instanceof long[]) {
                        l_arr = (long[]) v;
                        shape = new int[]{l_arr.length};
                    }
                    else if (v instanceof float[]) {
                        float[] fArr = (float[]) v;
                        d_arr = new double[fArr.length];
                        for (int i = 0; i < fArr.length; i++) {
                            d_arr[i] = fArr[i];
                        }
                        shape = new int[]{fArr.length};
                    }
                    else if (v instanceof double[]) {
                        d_arr = (double[]) v;
                        shape = new int[]{d_arr.length};
                    }
                    else if (v instanceof char[]) {
                        char[] cArr = (char[]) v;
                        i_arr = new int[cArr.length];
                        for (int i = 0; i < cArr.length; i++) {
                            i_arr[i] = cArr[i];
                        }
                        shape = new int[]{cArr.length};
                    }
                    else if (v instanceof byte[]) {
                        byte[] bArr = (byte[]) v;
                        i_arr = new int[bArr.length];
                        for (int i = 0; i < bArr.length; i++) {
                            i_arr[i] = bArr[i];
                        }
                        shape = new int[]{bArr.length};
                    }
                    else if (v instanceof short[]) {
                        short[] sArr = (short[]) v;
                        i_arr = new int[sArr.length];
                        for (int i = 0; i < sArr.length; i++) {
                            i_arr[i] = sArr[i];
                        }
                        shape = new int[]{sArr.length};
                    }
                    else {
                        log.warn("Unsupported primitive array property '{}': {}. Skipping.", key, v.getClass());
                        continue;
                    }
                }
                else if (v instanceof INDArray[]) {
                    INDArray[] arrArr = (INDArray[]) v;
                    List<Integer> validOffsets = new ArrayList<>();
                    for(INDArray arr : arrArr){
                        if(arr == null) {
                            validOffsets.add(0);
                            continue;
                        }
                        int rank = -1;
                        boolean skip = false;
                        try{
                            rank = arr.rank();
                            if(rank > Shape.MAX_RANK || rank < 0) {
                                skip=true;
                            }
                            if(!skip && arr.shape()!=null) {
                                for(long d : arr.shape()) {
                                    if(d>Integer.MAX_VALUE || d<0) {
                                        skip=true;
                                        break;
                                    }
                                }
                            }
                        } catch(Exception ex){
                            skip=true;
                        }
                        if(!skip){
                            try {
                                validOffsets.add(SameDiffSerializer.serializeSmallNdArrayToFlatBuffer(arr, fbb));
                            } catch(Exception serEx){
                                validOffsets.add(0);
                            }
                        } else {
                            validOffsets.add(0);
                        }
                    }
                    aIdx = Ints.toArray(validOffsets);
                    shape = new int[]{arrArr.length};
                }
                else if (v.getClass().getComponentType().isArray()) {
                    // Handle multi-dimensional primitive arrays
                    if (v instanceof int[][]) {
                        int[][] intArr2D = (int[][]) v;
                        List<Integer> flatList = new ArrayList<>();
                        for (int[] row : intArr2D) {
                            for (int val : row) {
                                flatList.add(val);
                            }
                        }
                        i_arr = Ints.toArray(flatList);
                        shape = new int[]{intArr2D.length, intArr2D[0].length};
                    }
                    else if (v instanceof double[][]) {
                        double[][] doubleArr2D = (double[][]) v;
                        List<Double> flatList = new ArrayList<>();
                        for (double[] row : doubleArr2D) {
                            for (double val : row) {
                                flatList.add(val);
                            }
                        }
                        d_arr = new double[flatList.size()];
                        for (int i = 0; i < flatList.size(); i++) {
                            d_arr[i] = flatList.get(i);
                        }
                        shape = new int[]{doubleArr2D.length, doubleArr2D[0].length};
                    }
                    else if (v instanceof long[][]) {
                        long[][] longArr2D = (long[][]) v;
                        List<Long> flatList = new ArrayList<>();
                        for (long[] row : longArr2D) {
                            for (long val : row) {
                                flatList.add(val);
                            }
                        }
                        l_arr = new long[flatList.size()];
                        for (int i = 0; i < flatList.size(); i++) {
                            l_arr[i] = flatList.get(i);
                        }
                        shape = new int[]{longArr2D.length, longArr2D[0].length};
                    }
                    else if (v instanceof boolean[][]) {
                        boolean[][] boolArr2D = (boolean[][]) v;
                        List<Boolean> flatList = new ArrayList<>();
                        for (boolean[] row : boolArr2D) {
                            for (boolean val : row) {
                                flatList.add(val);
                            }
                        }
                        b_arr = new boolean[flatList.size()];
                        for (int i = 0; i < flatList.size(); i++) {
                            b_arr[i] = flatList.get(i);
                        }
                        shape = new int[]{boolArr2D.length, boolArr2D[0].length};
                    }
                    else {
                        log.warn("Unsupported multi-dimensional array property '{}': {}. Skipping.", key, v.getClass());
                        continue;
                    }
                }
                else {
                    log.warn("Unsupported array property '{}': {}. Skipping.", key, v.getClass());
                    continue;
                }
            }
            else {
                log.warn("Unsupported property type '{}': {}. Skipping.", key, v.getClass());
                continue;
            }

            // Create vectors
            int idxD = FlatProperties.createDVector(fbb, d_arr != null ? d_arr : EMPTY_DOUBLE);
            int idxI = FlatProperties.createIVector(fbb, i_arr != null ? i_arr : EMPTY_INT);
            int idxL = FlatProperties.createLVector(fbb, l_arr != null ? l_arr : EMPTY_LONG);
            int idxA = FlatProperties.createAVector(fbb, aIdx != null ? aIdx : EMPTY_INT);
            int idxB = FlatProperties.createBVector(fbb, b_arr != null ? b_arr : EMPTY_BOOLEAN);
            int idxS = FlatProperties.createSVector(fbb, sIdx != null ? sIdx : EMPTY_INT);
            int idxShape = FlatProperties.createShapeVector(fbb, shape != null ? shape : EMPTY_INT);

            outOffsets.add(FlatProperties.createFlatProperties(fbb, iname, idxI, idxL, idxD, idxA, idxB, idxS, idxShape));
        } // End loop

        return Ints.toArray(outOffsets);
    }


    /** Clones a DifferentialFunction via FlatBuffers serialization/deserialization. */
    public static DifferentialFunction cloneViaSerialize(SameDiff sd, DifferentialFunction df, Map<String,Integer> nameToIdxMap, List<SDVariable> variableList ) throws IOException {
        Map<String,Integer> tempForwardMap = new HashMap<>(); // Use temp maps for clone context
        Map<String,Integer> tempFramesMap = new HashMap<>();
        AtomicInteger tempIdCounter = new AtomicInteger(0); // Independent ID counter for clone
        val bufferBuilder = new FlatBufferBuilder(1024); // Small buffer usually sufficient for single op

        // Pass true for isCloneContext to skip array serialization
        int fnOffset = FlatBuffersMapper.asFlatNode(sd, df, bufferBuilder, variableList, nameToIdxMap,
                tempForwardMap, tempFramesMap, tempIdCounter, 0, true); // ID 0 ok for clone
        bufferBuilder.finish(fnOffset);
        // Create duplicate to avoid buffer modification issues if builder is reused
        ByteBuffer cloneData = bufferBuilder.dataBuffer().duplicate();
        FlatNode flatNode = FlatNode.getRootAsFlatNode(cloneData);
        DifferentialFunction clone = FlatBuffersMapper.fromFlatNode(flatNode); // Deserialization doesn't need context flag
        return clone;
    }

    public static int[] mapOrNull(List<String> list, FlatBufferBuilder fbb) {
        if(list == null)
            return null;
        int[] out = new int[list.size()];
        int i = 0;
        for(String s : list) {
            out[i++] = fbb.createString(s);
        }
        return out;
    }

    @SneakyThrows // Propagates IOException from main method
    public static DifferentialFunction cloneViaSerialize(SameDiff sd, DifferentialFunction df) {
        Map<String,Integer> nameToIdxMap = new HashMap<>();
        List<SDVariable> variableList = new ArrayList<>(sd.variables()); // Use ordered list
        for (int i=0; i<variableList.size(); i++){
            if(variableList.get(i) != null && variableList.get(i).name() != null) {
                nameToIdxMap.put(variableList.get(i).name(), i);
            }
        }
        return cloneViaSerialize(sd, df, nameToIdxMap, variableList);
    }



    public static DifferentialFunction cloneViaSerialize(SameDiff sd, DifferentialFunction df, Map<String,Integer> nameToIdxMap) throws IOException {
        Map<String,Integer> temp2 = new HashMap<>();
        Map<String,Integer> temp3 = new HashMap<>();
        AtomicInteger temp4 = new AtomicInteger();

        val bufferBuilder = new FlatBufferBuilder(1024);
        int fn = FlatBuffersMapper.asFlatNode(sd, df, bufferBuilder,
                sd.variables(),
                nameToIdxMap,
                temp2,
                temp3,
                temp4,
                0);
        bufferBuilder.finish(fn);
        FlatNode flatNode = FlatNode.getRootAsFlatNode(bufferBuilder.dataBuffer());
        DifferentialFunction clone = FlatBuffersMapper.fromFlatNode(flatNode);
        return clone;
    }

    public static byte toVarType(VariableType variableType) {
        switch (variableType) {
            case VARIABLE:
                return VarType.VARIABLE;
            case CONSTANT:
                return VarType.CONSTANT;
            case ARRAY:
                return VarType.ARRAY;
            case PLACEHOLDER:
                return VarType.PLACEHOLDER;
            default:
                throw new RuntimeException("Unknown variable type: " + variableType);
        }
    }

    public static VariableType fromVarType(byte varType) {
        switch (varType) {
            case VarType.VARIABLE:
                return VariableType.VARIABLE;
            case VarType.CONSTANT:
                return VariableType.CONSTANT;
            case VarType.ARRAY:
                return VariableType.ARRAY;
            case VarType.PLACEHOLDER:
                return VariableType.PLACEHOLDER;
            default:
                throw new IllegalStateException("Unknown VarType byte value:" + varType);
        }
    }
}
