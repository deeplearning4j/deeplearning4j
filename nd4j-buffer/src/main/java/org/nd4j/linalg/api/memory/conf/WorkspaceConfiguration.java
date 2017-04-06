package org.nd4j.linalg.api.memory.conf;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.memory.enums.*;

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
    protected ResetPolicy policyReset;

    protected long initialSize;
    protected long minSize;
    protected long maxSize;

    protected int cyclesBeforeInitialization;

    protected double overallocationLimit;

    public static class WorkspaceConfigurationBuilder {
        private AllocationPolicy policyAllocation = AllocationPolicy.OVERALLOCATE;
        private SpillPolicy policySpill = SpillPolicy.EXTERNAL;
        private MirroringPolicy policyMirroring = MirroringPolicy.FULL;
        private LearningPolicy policyLearning = LearningPolicy.FIRST_LOOP;
        private ResetPolicy policyReset = ResetPolicy.BLOCK_LEFT;

        private long initialSize = 0;
        private long minSize = 0;
        private long maxSize = 0;

        protected int cyclesBeforeInitialization = 0;

        private double overallocationLimit = 0.3;
    }
}
