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

package org.nd4j.autodiff.samediff.serde;

import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.shade.guava.primitives.Ints;
import com.google.flatbuffers.FlatBufferBuilder;
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
     * This method converts enums for DataType
     */
    public static byte getDataTypeAsByte(@NonNull org.nd4j.linalg.api.buffer.DataType type) {
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
    public static org.nd4j.linalg.api.buffer.DataType getDataTypeFromByte(byte val) {
        if (val == DType.FLOAT) {
            return org.nd4j.linalg.api.buffer.DataType.FLOAT;
        } else if (val == DType.DOUBLE) {
            return org.nd4j.linalg.api.buffer.DataType.DOUBLE;
        } else if (val == DType.HALF) {
            return org.nd4j.linalg.api.buffer.DataType.HALF;
        } else if (val == DType.INT32) {
            return org.nd4j.linalg.api.buffer.DataType.INT;
        } else if (val == DType.INT64) {
            return org.nd4j.linalg.api.buffer.DataType.LONG;
        } else if (val == DType.INT8) {
            return org.nd4j.linalg.api.buffer.DataType.BYTE;
        } else if (val == DType.BOOL) {
            return org.nd4j.linalg.api.buffer.DataType.BOOL;
        } else if (val == DType.UINT8) {
            return org.nd4j.linalg.api.buffer.DataType.UBYTE;
        } else if (val == DType.INT16) {
            return org.nd4j.linalg.api.buffer.DataType.SHORT;
        } else if (val == DType.UTF8) {
            return org.nd4j.linalg.api.buffer.DataType.UTF8;
        } else if (val == DType.UINT16) {
            return org.nd4j.linalg.api.buffer.DataType.UINT16;
        } else if (val == DType.UINT32) {
            return org.nd4j.linalg.api.buffer.DataType.UINT32;
        } else if (val == DType.UINT64) {
            return org.nd4j.linalg.api.buffer.DataType.UINT64;
        } else if (val == DType.BFLOAT16){
            return org.nd4j.linalg.api.buffer.DataType.BFLOAT16;
        } else {
            throw new RuntimeException("Unknown datatype: " + val);
        }
    }


    /**
     * This method return operation ID for given op name/type pair.
     */
    public static long getOpNum(String name, Op.Type type) {
        if (type == Op.Type.LOOP) {
            return 0;
        } else if (type == Op.Type.RETURN) {
            return 40;
        } else if (type == Op.Type.CONDITIONAL) {
            return 10;
        } else if (type == Op.Type.LOOP_COND) {
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
        } else if (type == Op.Type.CUSTOM) {
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
            //return Nd4j.getExecutioner().getCustomOperations().get(name.toLowerCase()).getHash();

        } else {
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
    public static Op.Type getTypeFromByte(byte type) {
        switch (type) {
            case OpType.SCALAR:
                return Op.Type.SCALAR;
            case OpType.SCALAR_BOOL:
                return Op.Type.SCALAR_BOOL;
            case OpType.BROADCAST:
                return Op.Type.BROADCAST;
            case OpType.BROADCAST_BOOL:
                return Op.Type.BROADCAST_BOOL;
            case OpType.TRANSFORM_BOOL:
                return Op.Type.TRANSFORM_BOOL;
            case OpType.TRANSFORM_FLOAT:
                return Op.Type.TRANSFORM_FLOAT;
            case OpType.TRANSFORM_SAME:
                return Op.Type.TRANSFORM_SAME;
            case OpType.TRANSFORM_ANY:
                return Op.Type.TRANSFORM_ANY;
            case OpType.TRANSFORM_STRICT:
                return Op.Type.TRANSFORM_STRICT;
            case OpType.REDUCE_BOOL:
                return Op.Type.REDUCE_BOOL;
            case OpType.REDUCE_LONG:
                return Op.Type.REDUCE_LONG;
            case OpType.REDUCE_FLOAT:
                return Op.Type.REDUCE_FLOAT;
            case OpType.REDUCE_SAME:
                return Op.Type.REDUCE_SAME;
            case OpType.REDUCE_3:
                return Op.Type.REDUCE3;
            case OpType.INDEX_REDUCE:
                return Op.Type.INDEXREDUCE;
            case OpType.RANDOM:
                return Op.Type.RANDOM;
            case OpType.LOGIC:
                return Type.LOGIC;
            case OpType.CUSTOM:
                return Op.Type.CUSTOM;
            case OpType.PAIRWISE:
                return Op.Type.PAIRWISE;
            case OpType.PAIRWISE_BOOL:
                return Op.Type.PAIRWISE_BOOL;
            case OpType.SUMMARYSTATS:
                return Op.Type.SUMMARYSTATS;
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
    public static byte getFlatOpType(Op.Type type) {
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
        Op.Type opType = FlatBuffersMapper.getTypeFromByte(fn.opType());
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

        int[] dimensions = new int[fn.dimensionsLength()];
        for (int i = 0; i < dimensions.length; i++) {
            dimensions[i] = fn.dimensions(i);
        }
        FlatArray fa = fn.scalar();
        INDArray scalar = null;
        if (fa != null) {
            scalar = Nd4j.createFromFlatArray(fa);
        }

        FlatProperties[] flatProperties = new FlatProperties[fn.propertiesLength()];
        for (int i = 0; i < flatProperties.length; i++) {
            flatProperties[i] = fn.properties(i);
        }
        Map<String, Object> props = FlatBuffersMapper
                .mapFlatPropertiesToFunctionProperties(Arrays.asList(flatProperties));

        if (opType == Op.Type.CUSTOM || opType == Type.LOGIC) {
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
            //op.addTArgument();
            ((CustomOp) op).addIArgument(extraInteger);
            ((CustomOp) op).addTArgument(extraParams);
            ((CustomOp) op).addBArgument(extraBools);
            ((CustomOp) op).addDArgument(extraDTypes);

            op.setPropertiesForFunction(props);
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
            if (opType == Op.Type.SCALAR || opType == Op.Type.SCALAR_BOOL) {
                ScalarOp sOp = (ScalarOp) op;
                sOp.setScalar(scalar);
            } else if (opType == Op.Type.REDUCE_FLOAT || opType == Op.Type.REDUCE3 || opType == Op.Type.SUMMARYSTATS
                    || opType == Op.Type.VARIANCE
                    || opType == Op.Type.REDUCE_BOOL || opType == Op.Type.REDUCE_LONG
                    || opType == Op.Type.REDUCE_SAME) {
                val ba = (BaseReduceOp) op; //Reduce3 ops are also all BaseAccumulations
                ba.setDimensions(dimensions);
                ba.setDimensionz(Shape.ndArrayDimFromInt(dimensions));
            } else if (opType == Op.Type.INDEXREDUCE) {
                BaseIndexAccumulation bia = (BaseIndexAccumulation) op;
                bia.setDimensions(dimensions);
                bia.setDimensionz(Shape.ndArrayDimFromInt(dimensions));
            }
            /*
            Op types that don't need any extra/special mapping:
            TRANSFORM_BOOL - BooleanNot, IsFinite, IsInf, IsNaN, MatchConditionTransorm
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

    public static int[] mapFunctionPropertiesToFlatProperties(FlatBufferBuilder fbb, Map<String, Object> fnProps) {

        int[] outIdxs = new int[fnProps.size()];
        int count = 0;
        for (Map.Entry<String, Object> e : fnProps.entrySet()) {
            //Possible types here: primitives (as Number objects), primitive arrays, Strings, String arrays, multi-dimensional string/primitives
            Object v = e.getValue();
            int iname = fbb.createString(e.getKey());

            int[] i = null;
            long[] l = null;
            double[] d = null;
            int[] aIdx = null;
            boolean[] b = null;
            int[] sIdx = null;
            int[] shape = null;

            if (v == null) {
                //No op
            } else if (v instanceof Boolean) {
                b = new boolean[]{(Boolean) v};
            } else if(v instanceof Character){
                i = new int[]{(Character)v};
            } else if (v instanceof Number) {
                if (v instanceof Double) {
                    d = new double[]{(Double) v};
                } else if (v instanceof Float){
                    d = new double[]{(Float) v};
                } else if (v instanceof Integer) {
                    i = new int[]{(Integer) v};
                } else if (v instanceof Long) {
                    l = new long[]{(Long) v};
                } else {
                    throw new UnsupportedOperationException(
                            "Unable to map property \"" + e.getKey() + "\" of type " + v.getClass());
                }
            } else if (v instanceof String) {
                String str = (String) v;
                int strOffset = fbb.createString(str);
                sIdx = new int[]{strOffset};
            } else if (v instanceof org.nd4j.linalg.api.buffer.DataType) {
                String str = v.toString();
                int strOffset = fbb.createString(str);
                sIdx = new int[]{strOffset};
            } else if (v instanceof Enum) {
                String str = v.toString();
                int strOffset = fbb.createString(str);
                sIdx = new int[]{strOffset};
            } else if (v instanceof INDArray) {
                INDArray arr = (INDArray) v;
                aIdx = new int[]{arr.toFlatArray(fbb)};
            } else if (v.getClass().isArray()) {
                if (v.getClass().getComponentType().isPrimitive()) {
                    if (v instanceof boolean[]) {
                        b = (boolean[]) v;
                        shape = new int[]{b.length};
                    } else if (v instanceof double[]) {
                        d = (double[]) v;
                        shape = new int[]{d.length};
                    } else if (v instanceof int[]) {
                        i = (int[]) v;
                        shape = new int[]{i.length};
                    } else if (v instanceof long[]) {
                        l = (long[]) v;
                        shape = new int[]{l.length};
                    } else {
                        throw new UnsupportedOperationException(
                                "Unable to map property \"" + e.getKey() + "\" of type " + v.getClass());
                    }
                } else if (v instanceof String[]) {
                    //String[]
                    String[] strArr = (String[]) v;
                    sIdx = new int[strArr.length];
                    for (int j = 0; j < strArr.length; j++) {
                        sIdx[j] = fbb.createString(strArr[j]);
                    }
                    shape = new int[]{strArr.length};
                } else if (v instanceof INDArray[]) {
                    INDArray[] arrArr = (INDArray[]) v;
                    aIdx = new int[arrArr.length];
                    for (int j = 0; j < arrArr.length; j++) {
                        aIdx[j] = arrArr[j].toFlatArray(fbb);
                    }
                } else if (v.getClass().getComponentType().isArray()) {
                    shape = ArrayUtil.arrayShape(v, true);
                    //Multi-dimensional array
                    if (v instanceof boolean[][]) {
                        b = ArrayUtil.flatten((boolean[][]) v);
                    } else if (v instanceof boolean[][][]) {
                        b = ArrayUtil.flatten((boolean[][][]) v);
                    } else if (v instanceof double[][]) {
                        d = ArrayUtil.flatten((double[][]) v);
                    } else if (v instanceof double[][][]) {
                        d = ArrayUtil.flatten((double[][][]) v);
                    } else if (v instanceof int[][]) {
                        i = ArrayUtil.flatten((int[][]) v);
                    } else if (v instanceof int[][][]) {
                        i = ArrayUtil.flatten((int[][][]) v);
                    } else if (v instanceof long[][]) {
                        l = ArrayUtil.flatten((long[][]) v);
                    } else if (v instanceof long[][][]) {
                        l = ArrayUtil.flatten((long[][][]) v);
                    } else {
                        throw new UnsupportedOperationException(
                                "Unable to map multidimensional array property \"" + e.getKey() + "\" of type " + v
                                        .getClass());
                    }
                }
            }

            int idxD = FlatProperties.createDVector(fbb, d != null ? d : EMPTY_DOUBLE);
            int idxI = FlatProperties.createIVector(fbb, i != null ? i : EMPTY_INT);
            int idxL = FlatProperties.createLVector(fbb, l != null ? l : EMPTY_LONG);
            int idxA = FlatProperties.createAVector(fbb, aIdx != null ? aIdx : EMPTY_INT);
            int idxB = FlatProperties.createBVector(fbb, b != null ? b : EMPTY_BOOLEAN);
            int idxS = FlatProperties.createSVector(fbb, sIdx != null ? sIdx : EMPTY_INT);
            int idxShape = FlatProperties.createShapeVector(fbb, shape != null ? shape : EMPTY_INT);

            outIdxs[count++] = FlatProperties
                    .createFlatProperties(fbb, iname, idxI, idxL, idxD, idxA, idxB, idxS, idxShape);
        }
        return outIdxs;
    }

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
//                if(shape.length != 1){
//
//                    throw new IllegalStateException("Multi-dimensional arrays not yet implemented");
//                }

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
                } else if (p.sLength() > 0) {
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

    public static int asFlatNode(@NonNull SameDiff sameDiff, @NonNull DifferentialFunction node, @NonNull FlatBufferBuilder bufferBuilder, List<SDVariable> variables,
                             Map<String, Integer> reverseMap, Map<String, Integer> forwardMap, Map<String, Integer> framesMap, AtomicInteger idCounter, Integer id) {
        val opName = node.opName();
        val hash = FlatBuffersMapper.getOpNum(node.opName(), node.opType());
        //log.info("Exporting node: [{}:<{}> ; OpType: {}; Hash/opNum: {}]", node.opName(), node.tensorflowName(), node.opType(), hash);

        double[] extras;
        if (node.opType() == Op.Type.CUSTOM) {
            CustomOp op = (CustomOp) node;
            extras = op.tArgs();
        } else {
            Object[] eArgs = node.getExtraArgs();
            extras = eArgs != null ? new double[eArgs.length] : new double[0];
            for (int e = 0; e < extras.length; e++) {
                extras[e] = ((Number) eArgs[e]).doubleValue();
            }
        }

        boolean[] boolArgs = null;
        byte[] dtypeArgs = null;
        long[] extraBits = null;
        if (node.opType() == Op.Type.CUSTOM) {
            val dynamicCustomOp = (DynamicCustomOp) node;
            extraBits = dynamicCustomOp.iArgs();
            boolArgs = dynamicCustomOp.bArgs();

            if (dynamicCustomOp.numDArguments() > 0) {
                dtypeArgs = new byte[dynamicCustomOp.numDArguments()];
                val d = dynamicCustomOp.dArgs();
                for (int e = 0; e < dtypeArgs.length; e++) {
                    dtypeArgs[e] = (byte) d[e].toInt();
                }
            }
        } else if (node instanceof Enter) {
            // in case of Enter node we'll be storing unique frame reference
            val frameName = ((Enter) node).getFrameName();
            if (!framesMap.containsKey(frameName))
                framesMap.put(frameName, idCounter.incrementAndGet());

            extraBits = new long[]{framesMap.get(frameName).intValue()};
        } else
            extraBits = new long[]{};

        if (node.opType() == Op.Type.REDUCE_BOOL || node.opType() == Op.Type.REDUCE_SAME || node.opType() == Op.Type.REDUCE_FLOAT || node.opType() == Op.Type.REDUCE_LONG) {
            val op = (ReduceOp) node;

            boolArgs = new boolean[2];
            boolArgs[0] = op.isKeepDims();
            boolArgs[1] = true; // always new format
        } else if (node.opType() == Op.Type.INDEXREDUCE) {
            val op = (IndexAccumulation) node;

            boolArgs = new boolean[2];
            boolArgs[0] = op.isKeepDims();
            boolArgs[1] = true; // always new format
        }

        val inPaired = new ArrayList<Integer>();

        int[] outputIds = null;
        SDVariable[] outputVertexId = null;

        try {
            outputVertexId = node.outputVariables();
            outputIds = new int[outputVertexId.length];
            for (int i = 0; i < outputIds.length; i++) {
                outputIds[i] = variables.indexOf(outputVertexId[i]);
            }
        } catch (ND4UnresolvedOutputVariables e) {

            outputIds = new int[0];
            outputVertexId = null;
        } catch (Exception e) {
            throw new ND4JIllegalStateException(e);
        }


        SDVariable[] inputs = node.args();
        for (SDVariable input : inputs) {
            String varName = input.name();
            int outIdx;
            if (sameDiff.getVariables().get(varName).getOutputOfOp() != null) {
                DifferentialFunction df = sameDiff.getOps().get(sameDiff.getVariables().get(varName).getOutputOfOp()).getOp();
                outIdx = sameDiff.getOps().get(df.getOwnName()).getOutputsOfOp().indexOf(varName);
            } else {
                outIdx = 0;
            }

            if (!reverseMap.containsKey(varName)) {
                if (varName.contains("NextIteration")) {
                    // forward declaration: Merge node in case of loop will be referring to NextIteration node, which wasn't announced yet
                    int fwdNodeId = idCounter.incrementAndGet();
                    forwardMap.put(varName, fwdNodeId);
                    reverseMap.put(varName, fwdNodeId);
                } else {
                    throw new ND4JIllegalStateException("Unknown variable used in input: [" + varName + "]");
                }
            }

            int nodeId = reverseMap.get(varName);
            inPaired.add(IntPair.createIntPair(bufferBuilder, nodeId, outIdx));
        }

        log.trace("Own Name: {}", node.getOwnName());
        int ownId = id != null ? id : idCounter.incrementAndGet();  //forwardMap.containsKey(node.getOwnName()) ? forwardMap.get(node.getOwnName()) : idCounter.incrementAndGet();
        String[] outNames = node.outputVariablesNames();
        for (String s : outNames) {
            if (!reverseMap.containsKey(s)) {
                reverseMap.put(s, ownId);
            }
        }

        int[] dims;
        Type t = node.opType();
        if (t == Op.Type.REDUCE_FLOAT || t == Op.Type.REDUCE_SAME || t == Op.Type.REDUCE_BOOL
                || t == Op.Type.REDUCE_LONG || t == Op.Type.INDEXREDUCE || t == Op.Type.REDUCE3 || t == Type.VARIANCE || t == Type.SUMMARYSTATS) {
            dims = node.getDimensions();
            if (dims == null)
                dims = new int[0];
        } else {
            dims = new int[0];
        }

        Map<String, Object> fnProps = node.propertiesForFunction();
        int[] flatProperties = FlatBuffersMapper.mapFunctionPropertiesToFlatProperties(bufferBuilder, fnProps);
        int propIdx = FlatNode.createPropertiesVector(bufferBuilder, flatProperties);

        int nodesIn = FlatNode.createInputVector(bufferBuilder, new int[]{});
        int nodesInPaired = FlatNode.createInputPairedVector(bufferBuilder, Ints.toArray(inPaired));
        int nodesOut = FlatNode.createOutputVector(bufferBuilder, outputIds);
        int extraz = FlatNode.createExtraParamsVector(bufferBuilder, extras);
        int integerArgs = FlatNode.createExtraIntegerVector(bufferBuilder, extraBits);
        int bArgs = FlatNode.createExtraBoolsVector(bufferBuilder, boolArgs != null ? boolArgs : new boolean[0]);
        int dArgs = FlatNode.createOutputTypesVector(bufferBuilder, dtypeArgs != null ? dtypeArgs : new byte[0]);
        int dimensions = FlatNode.createDimensionsVector(bufferBuilder, dims);
        int fname = bufferBuilder.createString(node.getOwnName());
        int scopeName = bufferBuilder.createString("");
        int scalar = 0;
        if (node instanceof ScalarOp) {
            ScalarOp sOp = (ScalarOp) node;
            INDArray s = sOp.scalar();
            if (s != null) {
                scalar = s.toFlatArray(bufferBuilder);
            }
        }


        if (node.opType() == null)
            log.warn("Null-op node: {}", node);


        List<String> outVarNames = node.getSameDiff().getOps().get(node.getOwnName()).getOutputsOfOp();
        int[] outVarNamesStringsOffsets = new int[outVarNames == null ? 0 : outVarNames.size()];
        for (int i = 0; i < outVarNamesStringsOffsets.length; i++) {
            outVarNamesStringsOffsets[i] = bufferBuilder.createString(outVarNames.get(i));
        }
        int outVarNamesOffset = FlatNode.createOutputNamesVector(bufferBuilder, outVarNamesStringsOffsets);

        int opNameOffset = bufferBuilder.createString(opName);

        byte[] outTypes = new byte[outVarNames.size()];
        int i = 0;
        for (String s : outVarNames) {
            SDVariable v = sameDiff.getVariable(s);
            outTypes[i++] = FlatBuffersMapper.getDataTypeAsByte(v.dataType());
        }
        int outTypesOffset = FlatNode.createOutputTypesVector(bufferBuilder, outTypes);

        //Control dependencies:
        SameDiffOp sdo = sameDiff.getOps().get(node.getOwnName());

        int opCds = 0;
        int[] opCdsArr = mapOrNull(sdo.getControlDeps(), bufferBuilder);
        if(opCdsArr != null){
            opCds = FlatNode.createControlDepsVector(bufferBuilder, opCdsArr);
        }

        int varCds = 0;
        int[] varCdsArr = mapOrNull(sdo.getVarControlDeps(), bufferBuilder);
        if(varCdsArr != null){
            varCds = FlatNode.createVarControlDepsVector(bufferBuilder, varCdsArr);
        }

        int cdsFor = 0;
        int[] cdsForArr = mapOrNull(sdo.getControlDepFor(), bufferBuilder);
        if(cdsForArr != null){
            cdsFor = FlatNode.createControlDepForVector(bufferBuilder, cdsForArr);
        }


        int flatNode = FlatNode.createFlatNode(
                bufferBuilder,
                ownId,
                fname,
                FlatBuffersMapper.getFlatOpType(node.opType()),
                hash,
                propIdx,
                nodesIn,
                nodesInPaired,
                nodesOut,
                extraz,
                integerArgs,
                bArgs,
                dimensions,
                -1,     //Device
                0,      //Scope ID
                scopeName,      //Scope name
                outVarNamesOffset,
                opNameOffset,
                outTypesOffset,   //Output types
                scalar,
                opCds,
                varCds,
                cdsFor,
                dArgs
        );

        return flatNode;
    }

    public static int[] mapOrNull(List<String> list, FlatBufferBuilder fbb){
        if(list == null)
            return null;
        int[] out = new int[list.size()];
        int i=0;
        for(String s : list){
            out[i++] = fbb.createString(s);
        }
        return out;
    }

    public static DifferentialFunction cloneViaSerialize(SameDiff sd, DifferentialFunction df ){
        Map<String,Integer> nameToIdxMap = new HashMap<>();
        int count = 0;
        for( Variable v : sd.getVariables().values()){
            nameToIdxMap.put(v.getName(), count++);
        }
        return cloneViaSerialize(sd, df, nameToIdxMap);
    }

    public static DifferentialFunction cloneViaSerialize(SameDiff sd, DifferentialFunction df, Map<String,Integer> nameToIdxMap ){
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
