package org.nd4j.autodiff.samediff.serde;

import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.base.Preconditions;
import org.nd4j.graph.*;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.impl.accum.BaseReduction;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.ByteOrder;

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

    public static Class<?> getOpClass(long idHash, Op.Type type){
        switch (type){
            case CUSTOM:
                return DifferentialFunctionClassHolder.getInstance().customOpClassForHash(idHash);
            case SCALAR:
            case TRANSFORM:
            case PAIRWISE:
            case SPECIAL:
            case BROADCAST:
            case REDUCE:
            case INDEXREDUCE:
            case VARIANCE:
            case REDUCE3:
            case RANDOM:
                return LegacyOpMapper.getLegacyOpClassForId(type, (int)idHash);

            case LOOP:
            case RETURN:
            case IF:
            case CONDITIONAL:
            case MERGE:
            case LOOP_COND:
            case NEXT_ITERATION:
            case EXIT:
            case ENTER:
            default:
                throw new UnsupportedOperationException("Not supported or not implemneted: op type " + type + ", id/hash " + idHash );
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
        FlatProperties properties = fn.propertiesLength() > 0 ? fn.properties(0) : null;
        int[] input = new int[fn.inputLength()];
        for( int i=0; i<input.length; i++ ){
            input[i] = fn.input(i);
        }
        IntPair[] inputPaired = new IntPair[fn.inputPairedLength()];
        for( int i=0; i<inputPaired.length; i++ ){
            inputPaired[i] = fn.inputPaired(i);
        }
//            DataBuffer.Type dt = SameDiff.getDataTypeFromByte(fn.dataType());
        int[] output = new int[fn.outputLength()];
        for( int i=0; i<output.length; i++ ){
            output[i] = fn.output(i);
        }
//        double[] extraParams = new double[fn.extraParamsLength()];
        Object[] extraParams = new Object[fn.extraParamsLength()];
        for( int i=0; i<extraParams.length; i++ ){
            extraParams[i] = fn.extraParams(i);
        }
        long[] extraInteger = new long[fn.extraIntegerLength()];
        for( int i=0; i<extraInteger.length; i++ ){
            extraInteger[i] = fn.extraInteger(i);
        }
        int[] dimensions = new int[fn.dimensionsLength()];
        for( int i=0; i<dimensions.length; i++ ){
            dimensions[i] = fn.dimensions(i);
        }
        float scalar = fn.scalar();

        if(opType == Op.Type.CUSTOM) {
//            DifferentialFunction df = DifferentialFunctionClassHolder.getInstance().getInstance(name);
            Class<?> c = DifferentialFunctionClassHolder.getInstance().customOpClassForHash(opNum);

            Preconditions.checkNotNull(c, "Could not find class for hash %s", opNum);

            DifferentialFunction op;
            try {
                op = (DifferentialFunction) c.newInstance();
            } catch (IllegalAccessException | InstantiationException e) {
                throw new RuntimeException("Error creating differential function instance of type " + c);
            }

            //Set input SDVariables:

            //Set args:
            //op.addTArgument();
            if (op instanceof CustomOp) {
                //TODO where are T args?
                ((CustomOp) op).addIArgument(extraInteger);
            }

            return op;
        } else { //if (opType == Op.Type.SCALAR || opType == Op.Type.TRANSFORM) {
            Class<?> c = LegacyOpMapper.getLegacyOpClassForId(opType, (int)opNum);
            Op op;
            try {
                op = (Op) c.newInstance();
            } catch (IllegalAccessException | InstantiationException e) {
                throw new RuntimeException("Error creating differential function (Op) instance of type " + c);
            }

            op.setExtraArgs(extraParams);
            if(opType == Op.Type.SCALAR){
                ScalarOp sOp = (ScalarOp)op;
                sOp.setScalar(scalar);
            } else if(opType == Op.Type.REDUCE){
                BaseAccumulation ba = (BaseAccumulation)op;
                ba.setDimensions(dimensions);
                ba.setNewFormat(true);  //Always "new" format (i.e., rank 0 scalars, not rank 2) for SameDiff-based exec
            } else if(opType == Op.Type.INDEXREDUCE){
                BaseIndexAccumulation bia = (BaseIndexAccumulation)op;
                bia.setDimensions(dimensions);
                bia.setNewFormat(true);  //Always "new" format (i.e., rank 0 scalars, not rank 2) for SameDiff-based exec
            }
            return (DifferentialFunction)op;
        }
//        else {
//            throw new UnsupportedOperationException("Not yet implemented: op type " + opType);
//        }

    }
}
