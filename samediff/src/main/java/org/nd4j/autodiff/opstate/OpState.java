package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.autodiff.ArrayField;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

/**
 * Describes the type of
 * operation that needs to happen
 * @author Adam Gibson
 */
@Data
@Builder
@EqualsAndHashCode
public class OpState implements Serializable {
    private long n;
    private OpType opType;
    private String opName;
    private int opNum;
    private Number scalarValue;
    private String[] vertexIds;
    private String id;
    private int[] axes;
    private Object[] extraArgs;
    private Integer[] extraBits;
    private Object[] extraArgsWithoutInPlace;
    private NDArrayInformation result;
    //function handle mainly used for autodiff invocation
    private DifferentialFunction differentialFunction;
    private ArrayField arrayField;
    private boolean inPlace;

    /**
     * Creates an op state from
     * the given op.
     * @param op
     * @param arrToVertexID a map of {@link INDArray}
     *                      to vertex id (this map
     *                      is typically a reference map
     *                      {@link java.util.IdentityHashMap})
     * @return
     */
    public static OpState fromOp(Op op, Map<INDArray,Integer> arrToVertexID) {
        OpState opState = OpState.builder()
                .extraArgs(op.extraArgs())
                .opNum(op.opNum())
                .n(op.n()).vertexIds(null)
                .id(UUID.randomUUID().toString())
                .opName(op.name()).vertexIds(new String[]{
                        String.valueOf(arrToVertexID.get(op.x()))
                        ,String.valueOf(arrToVertexID.get(op.y()))
                })
                .opType(opTypeFromOp(op))
                .build();
        NDArrayInformation ndArrayInformation = NDArrayInformation.newInfo(op.z());
        ndArrayInformation.setOwner(opState);
        opState.setResult(ndArrayInformation);
        return opState;
    }


    /**
     * Create an {@link OpType}
     * based on the type of {@link Op}
     * @param op the input op
     * @return the optype based on
     * the given op
     */
    public static OpType opTypeFromOp(Op op) {
       if(op instanceof ScalarOp)
           return OpType.SCALAR_TRANSFORM;
       else if(op instanceof Accumulation)
           return OpType.ACCUMULATION;
       else if(op instanceof IndexAccumulation)
           return OpType.INDEX_ACCUMULATION;
       else if(op instanceof GridOp)
           return OpType.AGGREGATE;
       else if(op instanceof TransformOp)
           return OpType.TRANSFORM;
       else if(op instanceof ShapeOp)
           return OpType.SHAPE;
       else if(op instanceof BroadcastOp)
           return OpType.BROADCAST;
       else if (op instanceof CustomOp)
           return OpType.CUSTOM;
       throw new IllegalStateException("Illegal op type " + op.getClass().getName());
    }

    /**
     *
     * @return
     */
    public boolean isInPlace() {
        return inPlace;
    }

    /**
     *
     * @return
     */
    public Object[] getExtraArgs() {
        if(extraArgs == null || extraArgs.length <= 0)
            return null;
        if(extraArgsWithoutInPlace == null || extraArgsWithoutInPlace.length <= 0) {
            extraArgsWithoutInPlace = new Object[extraArgs.length > 1 ? extraArgs.length : 1];
            int count = 0;
            for(int i = 0; i < extraArgs.length; i++) {
                if(!(extraArgs[i] instanceof Boolean))
                    extraArgsWithoutInPlace[count++] = extraArgs[i];
            }
        }
        return extraArgsWithoutInPlace;
    }

    public void setExtraArgs(Object[] extraArgs) {
        this.extraArgs = extraArgs;
    }



    public  enum OpType {
        SCALAR_TRANSFORM,
        ACCUMULATION,
        TRANSFORM,
        BROADCAST,
        INDEX_ACCUMULATION,
        AGGREGATE,
        SHAPE,
        GRADIENT,
        CUSTOM,
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        OpState opState = (OpState) o;

        if (n != opState.n) return false;
        if (opType != opState.opType) return false;
        if (opName != null ? !opName.equals(opState.opName) : opState.opName != null) return false;
        if (scalarValue != null ? !scalarValue.equals(opState.scalarValue) : opState.scalarValue != null) return false;
        // Probably incorrect - comparing Object[] arrays with Arrays.equals
        if (!Arrays.equals(vertexIds, opState.vertexIds)) return false;
        if (id != null ? !id.equals(opState.id) : opState.id != null) return false;
        if (!Arrays.equals(axes, opState.axes)) return false;
        // Probably incorrect - comparing Object[] arrays with Arrays.equals
        if(extraArgs != null && opState.extraArgs != null)
            if(extraArgs.length != opState.extraArgs.length)
                return false;
        if (result != null ? !result.equals(opState.result) : opState.result != null) return false;
        return true;
    }

    @Override
    public int hashCode() {
        int result1 = super.hashCode();
        result1 = 31 * result1 + (int) (n ^ (n >>> 32));
        result1 = 31 * result1 + (opType != null ? opType.hashCode() : 0);
        result1 = 31 * result1 + (opName != null ? opName.hashCode() : 0);
        result1 = 31 * result1 + (scalarValue != null ? scalarValue.hashCode() : 0);
        result1 = 31 * result1 + Arrays.hashCode(vertexIds);
        result1 = 31 * result1 + (id != null ? id.hashCode() : 0);
        result1 = 31 * result1 + Arrays.hashCode(axes);
        result1 = 31 * result1 + Arrays.hashCode(extraArgs);
        result1 = 31 * result1 + Arrays.hashCode(extraArgsWithoutInPlace);
        result1 = 31 * result1 + (result != null ? result.hashCode() : 0);
        return result1;
    }
}
