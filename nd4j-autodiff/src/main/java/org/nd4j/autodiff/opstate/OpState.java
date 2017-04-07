package org.nd4j.autodiff.opstate;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;

import java.io.Serializable;

/**
 * Created by agibsonccc on 4/6/17.
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

    public  enum OpType {
        SCALAR_TRANSFORM,
        ACCUMULATION,
        TRANSFORM,
        BROADCAST,
        INDEX_ACCUMULATION,
        AGGREGATE
    }

}
