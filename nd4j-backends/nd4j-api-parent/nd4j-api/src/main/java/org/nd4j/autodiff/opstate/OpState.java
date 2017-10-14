package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.linalg.api.ops.Op;

import java.io.Serializable;
import java.util.Arrays;

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
    private Op.Type opType;
    private String opName;
    private int opNum;
    private Number scalarValue;
    private String[] vertexIds;
    private String id;
    private int[] axes;
    private Object[] extraArgs;
    private int[] extraBits;
    private Object[] extraArgsWithoutInPlace;
    private NDArrayInformation[] results;
    //function handle mainly used for autodiff invocation
    private DifferentialFunction differentialFunction;
    private boolean inPlace;

     OpState(long n, Op.Type opType, String opName, int opNum, Number scalarValue, String[] vertexIds, String id, int[] axes, Object[] extraArgs, int[] extraBits, Object[] extraArgsWithoutInPlace, NDArrayInformation[] results, DifferentialFunction differentialFunction, boolean inPlace) {
        this.n = n;
        this.opType = opType;
        this.opName = opName;
        this.opNum = opNum;
        this.scalarValue = scalarValue;
        this.vertexIds = vertexIds;
        this.id = id;
        this.axes = axes;
        this.extraArgs = extraArgs;
        this.extraBits = extraBits;
        this.extraArgsWithoutInPlace = extraArgsWithoutInPlace;
        this.results = results;
        this.differentialFunction = differentialFunction;
        this.inPlace = inPlace;
    }

    public DifferentialFunction getDifferentialFunction() {
        if(differentialFunction != null)
            return differentialFunction.getSameDiff().setupFunction(differentialFunction);
        return null;
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
        if (results != null ? !Arrays.equals(results,opState.results) : opState.results != null) return false;
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
        result1 = 31 * result1 + (results != null ? Arrays.hashCode(results) : 0);
        return result1;
    }
}
