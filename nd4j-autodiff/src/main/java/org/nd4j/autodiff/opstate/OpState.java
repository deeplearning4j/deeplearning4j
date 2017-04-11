package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;

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
    private NDArrayInformation result;

    public  enum OpType {
        SCALAR_TRANSFORM,
        ACCUMULATION,
        TRANSFORM,
        BROADCAST,
        INDEX_ACCUMULATION,
        AGGREGATE
    }

}
