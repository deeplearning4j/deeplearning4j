package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

/**
 * Describes the type of operation that needs to happen
 * @author Adam Gibson
 */
@Data
@Builder
@EqualsAndHashCode
public class OpState implements Serializable {
    private long n;
    private OpType opType;
    private String opName;
    private Number scalarValue;
    private String[] vertexIds;
    private String id;
    private int[] axes;
    private Object[] extraArgs;
    private Object[] extraArgsWithoutInPlace;
    private NDArrayInformation result;


    public boolean isInPlace() {
        return getInPlace(extraArgs);
    }

    public Object[] getExtraArgs() {
        if(extraArgs == null)
            return null;
        if(extraArgsWithoutInPlace == null) {
            extraArgsWithoutInPlace = new Object[extraArgs.length - 1];
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

    protected boolean getInPlace(Object[] extraArgs) {
        if(extraArgs == null) {
            return false;
        }
        else {
            for(int i = 0; i < extraArgs.length; i++) {
                if(extraArgs[i] instanceof Boolean)
                    return (Boolean) extraArgs[i];
            }
        }

        return false;
    }

    public  enum OpType {
        SCALAR_TRANSFORM,
        ACCUMULATION,
        TRANSFORM,
        BROADCAST,
        INDEX_ACCUMULATION,
        AGGREGATE,
        SHAPE
    }



}
