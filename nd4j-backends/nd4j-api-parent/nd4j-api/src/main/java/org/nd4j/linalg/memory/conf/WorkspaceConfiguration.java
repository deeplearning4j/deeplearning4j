package org.nd4j.linalg.memory.conf;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.memory.MemoryManager;
import org.nd4j.linalg.memory.enums.AllocationPolicy;
import org.nd4j.linalg.memory.enums.LearningPolicy;
import org.nd4j.linalg.memory.enums.MirroringPolicy;
import org.nd4j.linalg.memory.enums.SpillPolicy;

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicLong;

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
