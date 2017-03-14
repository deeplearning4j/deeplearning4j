package org.nd4j.linalg.api.memory.conf;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;

import java.io.Serializable;

/**
 * @author raver119@gmail.com
 */
@Builder
@Data
@NoArgsConstructor
@AllArgsConstructor
// TODO: add json mapping here
public class WorkspaceConfiguration implements Serializable {
    protected AllocationPolicy policyAllocation;
    protected SpillPolicy policySpill;
    protected MirroringPolicy policyMirroring;
    protected LearningPolicy policyLearning;

    protected long initialSize;
    protected long maxSize;

    protected double overallocationLimit;
}
