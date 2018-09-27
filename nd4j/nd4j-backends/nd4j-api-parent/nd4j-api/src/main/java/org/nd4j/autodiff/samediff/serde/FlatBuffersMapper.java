package org.nd4j.autodiff.samediff.serde;

import com.google.flatbuffers.FlatBufferBuilder;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.base.Preconditions;
import org.nd4j.graph.*;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.accum.BaseReduction;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.nio.ByteOrder;
import java.util.*;

public class FlatBuffersMapper {

    private FlatBuffersMapper(){ }

    /**
     * This method converts enums for DataType
     *
     * @param type
     * @return
     */
    public static byte getDataTypeAsByte(DataBuffer.Type type) {
        switch (type) {
            case FLOAT:
                return DataType.FLOAT;
            case DOUBLE:
                return DataType.DOUBLE;
            case HALF:
                return DataType.HALF;
            case INT:
                return DataType.INT32;
            case LONG:
                return DataType.INT64;
            default:
                throw new ND4JIllegalStateException("Unknown or unsupported DataType used: [" + type + "]");
        }
    }

    /**
     * This method converts enums for DataType
     *
     * @param val
     * @return
     */
    public static DataBuffer.Type getDataTypeFromByte(byte val) {
        if (val == DataType.FLOAT)
            return DataBuffer.Type.FLOAT;
        else if (val == DataType.DOUBLE)
            return DataBuffer.Type.DOUBLE;
        else if (val == DataType.HALF)
            return DataBuffer.Type.HALF;

        throw new UnsupportedOperationException("Unsupported DataType: [" + val + "]");
    }




    /**
     * This method return operation ID for given op name/type pair.
     *
     * @param name
     * @param type
     * @return
     */
    public static long getOpNum(String name, Op.Type type) {
        if (type == Op.Type.LOOP) {
            return 0;
        } else if (type == Op.Type.RETURN) {
            return 40;
        } else if (type == Op.Type.IF) {
            return 30;
        } else if (type == Op.Type.CONDITIONAL) {
            return 10;
        } else if (type == Op.Type.MERGE) {
            return 60L;
        } else if (type == Op.Type.LOOP_COND) {
            return 70L;
        } else if (type == Op.Type.NEXT_ITERATION) {
            return 80L;
        } else if (type == Op.Type.EXIT) {
            return 90L;
        } else if (type == Op.Type.ENTER) {
            return 100L;
        } else if (type == Op.Type.CUSTOM) {
            val name2 = Nd4j.getExecutioner().getCustomOperations().get(name.toLowerCase());
            if (name2 == null) {
                val name3 = Nd4j.getExecutioner().getCustomOperations().get(name);
                if (name3 == null)
                    return 0;
                else
                    return name3.getHash();
            } else
                return name2.getHash();
            //return Nd4j.getExecutioner().getCustomOperations().get(name.toLowerCase()).getHash();

        } else
            return (long) Nd4j.getOpFactory().getOpNumByName(name);
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
            case OpType.BROADCAST:
                return Op.Type.BROADCAST;
            case OpType.TRANSFORM:
                return Op.Type.TRANSFORM;
            case OpType.ACCUMULATION:
                return Op.Type.REDUCE;
            case OpType.ACCUMULATION3:
                return Op.Type.REDUCE3;
            case OpType.INDEX_ACCUMULATION:
                return Op.Type.INDEXREDUCE;
            case OpType.RANDOM:
                return Op.Type.RANDOM;
            case OpType.LOGIC:
                return Op.Type.META;
            case OpType.CUSTOM:
                return Op.Type.CUSTOM;
            case OpType.SHAPE:
                return Op.Type.SHAPE;
            case OpType.PAIRWISE:
                return Op.Type.PAIRWISE;
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
            case BROADCAST:
                return OpType.BROADCAST;
            case TRANSFORM:
            case SPECIAL:
                return OpType.TRANSFORM;
            case REDUCE:
                return OpType.ACCUMULATION;
            case REDUCE3:
                return OpType.ACCUMULATION3;
            case INDEXREDUCE:
                return OpType.INDEX_ACCUMULATION;
            case RANDOM:
                return OpType.RANDOM;
            case VARIANCE:
                return OpType.SUMMARYSTATS;
            case MERGE:
            case CONDITIONAL:
            case LOOP:
            case RETURN:
            case ENTER:
            case EXIT:
            case NEXT_ITERATION:
            case LOOP_COND:
            case IF:
                return OpType.LOGIC;
            case CUSTOM:
                return OpType.CUSTOM;
            case SHAPE:
                return OpType.SHAPE;
            case PAIRWISE:
                return OpType.PAIRWISE;
            case SUMMARYSTATS:
                return OpType.SUMMARYSTATS;
            default:
                throw new UnsupportedOperationException("Unknown op type passed in: " + type);
        }
    }


    /**
     * This method just converts enums
     *
     * @param val
     * @return
     */
    public static ByteOrder getOrderFromByte(byte val) {
        if (val == org.nd4j.graph.ByteOrder.LE)
            return ByteOrder.LITTLE_ENDIAN;
        else
            return ByteOrder.BIG_ENDIAN;
    }

    /**
     * This method returns current byte order for this JVM as libnd4j enum
     *
     * @return
     */
    public static byte getOrderAsByte() {
        if (ByteOrder.nativeOrder().equals(ByteOrder.BIG_ENDIAN))
            return org.nd4j.graph.ByteOrder.BE;
        else
            return org.nd4j.graph.ByteOrder.LE;
    }

    public static DifferentialFunction fromFlatNode(FlatNode fn){

        int id = fn.id();               //ID of the node
        String name = fn.name();        //Name of the node, NOT the name of the op
        Op.Type opType = FlatBuffersMapper.getTypeFromByte(fn.opType());
        long opNum = fn.opNum();        //Op num: hash for custom, number for legacy
        int[] input = new int[fn.inputLength()];
        for( int i=0; i<input.length; i++ ){
            input[i] = fn.input(i);
        }
        IntPair[] inputPaired = new IntPair[fn.inputPairedLength()];
        for( int i=0; i<inputPaired.length; i++ ){
            inputPaired[i] = fn.inputPaired(i);
        }
        int[] output = new int[fn.outputLength()];
        for( int i=0; i<output.length; i++ ){
            output[i] = fn.output(i);
        }
        double[] extraParams = new double[fn.extraParamsLength()];
        for( int i=0; i<extraParams.length; i++ ){
            extraParams[i] = fn.extraParams(i);
        }
//        Object[] extraParams = new Object[fn.extraParamsLength()];
//        for( int i=0; i<extraParams.length; i++ ){
//            extraParams[i] = fn.extraParams(i);
//        }
        long[] extraInteger = new long[fn.extraIntegerLength()];
        for( int i=0; i<extraInteger.length; i++ ){
            extraInteger[i] = fn.extraInteger(i);
        }
        int[] dimensions = new int[fn.dimensionsLength()];
        for( int i=0; i<dimensions.length; i++ ){
            dimensions[i] = fn.dimensions(i);
        }
        float scalar = fn.scalar();

        FlatProperties[] flatProperties = new FlatProperties[fn.propertiesLength()];
        for( int i=0; i<flatProperties.length; i++ ){
            flatProperties[i] = fn.properties(i);
        }
        Map<String,Object> props = FlatBuffersMapper.mapFlatPropertiesToFunctionProperties(Arrays.asList(flatProperties));


        if(opType == Op.Type.CUSTOM) {
            String opName = fn.opName();
            Class<?> c = DifferentialFunctionClassHolder.getInstance().customOpClassForHashAndName(opNum, opName);

            Preconditions.checkNotNull(c, "Could not find class for hash %s", opNum);

            DifferentialFunction op;
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

            op.setPropertiesForFunction(props);
            return op;
        } else {
            Class<?> c = LegacyOpMapper.getLegacyOpClassForId(opType, (int)opNum);
            Op op;
            try {
                op = (Op) c.newInstance();
            } catch (IllegalAccessException | InstantiationException e) {
                throw new RuntimeException("Error creating differential function (Op) instance of type " + c);
            }

            Object[] extraParamsObj = new Object[extraParams.length];
            for( int i=0; i<extraParams.length; i++ ){
                extraParamsObj[i] = extraParams[i];
            }
            op.setExtraArgs(extraParamsObj);
            if(opType == Op.Type.SCALAR){
                ScalarOp sOp = (ScalarOp)op;
                sOp.setScalar(scalar);
            } else if(opType == Op.Type.REDUCE || opType == Op.Type.REDUCE3 || opType == Op.Type.SUMMARYSTATS || opType == Op.Type.VARIANCE){
                BaseAccumulation ba = (BaseAccumulation)op; //Reduce3 ops are also all BaseAccumulations
                ba.setDimensions(dimensions);
                ba.setNewFormat(true);  //Always "new" format (i.e., rank 0 scalars, not rank 2) for SameDiff-based exec
            } else if(opType == Op.Type.INDEXREDUCE){
                BaseIndexAccumulation bia = (BaseIndexAccumulation)op;
                bia.setDimensions(dimensions);
                bia.setNewFormat(true);  //Always "new" format (i.e., rank 0 scalars, not rank 2) for SameDiff-based exec
            }

            ((DifferentialFunction)op).setPropertiesForFunction(props);
            return (DifferentialFunction)op;
        }
    }

    private static final boolean[] EMPTY_BOOLEAN = new boolean[0];
    private static final int[] EMPTY_INT = new int[0];
    private static final long[] EMPTY_LONG = new long[0];
    private static final double[] EMPTY_DOUBLE = new double[0];

    public static int[] mapFunctionPropertiesToFlatProperties(FlatBufferBuilder fbb, Map<String,Object> fnProps){

        int[] outIdxs = new int[fnProps.size()];
        int count = 0;
        for(Map.Entry<String,Object> e : fnProps.entrySet()){
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



            if(v == null) {
                //No op
            } else if(v instanceof Boolean){
                b = new boolean[]{(Boolean)v};
            } else if(v instanceof Number) {
                if (v instanceof Double) {
                    d = new double[]{(Double) v};
                } else if (v instanceof Integer) {
                    i = new int[]{(Integer) v};
                } else if (v instanceof Long) {
                    l = new long[]{(Long) v};
                } else {
                    throw new UnsupportedOperationException("Unable to map property \"" + e.getKey() + "\" of type " + v.getClass());
                }
            } else if(v instanceof String) {
                String str = (String) v;
                int strOffset = fbb.createString(str);
                sIdx = new int[]{strOffset};
            } else if(v instanceof INDArray){
                INDArray arr = (INDArray)v;
                aIdx = new int[]{arr.toFlatArray(fbb)};
            } else if(v.getClass().isArray()){
                if(v.getClass().getComponentType().isPrimitive()){
                    if(v instanceof boolean[]) {
                        b = (boolean[])v;
                        shape = new int[]{b.length};
                    } else if(v instanceof double[]){
                        d = (double[])v;
                        shape = new int[]{d.length};
                    } else if(v instanceof int[]){
                        i = (int[])v;
                        shape = new int[]{i.length};
                    } else if(v instanceof long[]){
                        l = (long[])v;
                        shape = new int[]{l.length};
                    } else {
                        throw new UnsupportedOperationException("Unable to map property \"" + e.getKey() + "\" of type " + v.getClass());
                    }
                } else if (v instanceof String[]) {
                    //String[]
                    String[] strArr = (String[]) v;
                    sIdx = new int[strArr.length];
                    for (int j = 0; j < strArr.length; j++) {
                        sIdx[j] = fbb.createString(strArr[j]);
                    }
                    shape = new int[]{strArr.length};
                } else if (v instanceof INDArray[]){
                    INDArray[] arrArr = (INDArray[])v;
                    aIdx = new int[arrArr.length];
                    for( int j=0; j<arrArr.length; j++){
                        aIdx[j] = arrArr[j].toFlatArray(fbb);
                    }
                } else if(v.getClass().getComponentType().isArray()){
                    //Multi-dimensional array
                    throw new UnsupportedOperationException("Multi-dimensional array not yet implemented");
                }
            }

            int idxD = FlatProperties.createDVector(fbb, d != null ? d : EMPTY_DOUBLE);
            int idxI = FlatProperties.createIVector(fbb, i != null ? i : EMPTY_INT);
            int idxL = FlatProperties.createLVector(fbb, l != null ? l : EMPTY_LONG);
            int idxA = FlatProperties.createAVector(fbb, aIdx != null ? aIdx : EMPTY_INT);
            int idxB = FlatProperties.createBVector(fbb, b != null ? b : EMPTY_BOOLEAN);
            int idxS = FlatProperties.createSVector(fbb, sIdx != null ? sIdx : EMPTY_INT);
            int idxShape = FlatProperties.createShapeVector(fbb, shape != null ? shape : EMPTY_INT);

            outIdxs[count++] = FlatProperties.createFlatProperties(fbb, iname, idxI, idxL, idxD, idxA, idxB, idxS, idxShape);

        }
        return outIdxs;
    }

    public static Map<String,Object> mapFlatPropertiesToFunctionProperties(Iterable<FlatProperties> list){
        Map<String,Object> out = new HashMap<>();
        for(FlatProperties p : list){

            String name = p.name();
            //Work out type:
            if(p.shapeLength() > 0){
                //Array type
                int[] shape = new int[p.shapeLength()];
                for( int i=0; i<shape.length; i++ ){
                    shape[i] = p.shape(i);
                }
                if(shape.length != 1){
                    throw new IllegalStateException("Multi-dimensional arrays not yet implemented");
                }

                if(p.iLength() > 0){
                    int[] iArr = new int[p.iLength()];
                    for( int i=0; i<iArr.length; i++ ){
                        iArr[i] = p.i(i);
                    }
                    out.put(name, iArr);
                } else if(p.dLength() > 0){
                    double[] dArr = new double[p.dLength()];
                    for( int i=0; i<dArr.length; i++ ){
                        dArr[i] = p.d(i);
                    }
                    out.put(name, dArr);
                } else if(p.lLength() > 0) {
                    long[] lArr = new long[p.lLength()];
                    for (int i = 0; i < lArr.length; i++) {
                        lArr[i] = p.l(i);
                    }
                    out.put(name, lArr);
                } else if(p.bLength() > 0){
                    boolean[] bArr = new boolean[p.bLength()];
                    for( int i=0; i<bArr.length; i++ ){
                        bArr[i] = p.b(i);
                    }
                    out.put(name, bArr);
                } else if(p.sLength() > 0){
                    String[] sArr = new String[p.sLength()];
                    for( int i=0; i<sArr.length; i++ ){
                        sArr[i] = p.s(i);
                    }
                    out.put(name, sArr);
                } else if(p.aLength() > 0){
                    INDArray[] iArr = new INDArray[p.aLength()];
                    for( int i=0; i<iArr.length; i++ ){
                        FlatArray fa = p.a(0);
                        iArr[i] = Nd4j.createFromFlatArray(fa);
                    }
                    out.put(name, iArr);
                }  else {
                    //null property case
                    out.put(name, null);
                }
            } else {
                //non-array primitive, String or INDArray
                if(p.bLength() > 0) {
                    out.put(name, p.b(0));
                } else if(p.iLength() > 0){
                    out.put(name, p.i(0));
                } else if(p.lLength() > 0){
                    out.put(name, p.l(0));
                } else if(p.dLength() > 0){
                    out.put(name, p.d(0));
                } else if(p.sLength() > 0){
                    out.put(name, p.s(0));
                } else if(p.aLength() > 0){
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
}
