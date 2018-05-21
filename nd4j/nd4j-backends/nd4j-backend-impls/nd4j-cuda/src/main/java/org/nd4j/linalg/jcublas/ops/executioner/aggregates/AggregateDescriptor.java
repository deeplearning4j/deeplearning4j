package org.nd4j.linalg.jcublas.ops.executioner.aggregates;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;

/**
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class AggregateDescriptor {
    private Aggregate op;
    private long aggregationKey;
    private long index;
}
