package org.nd4j.linalg.api.memory.conf;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.memory.enums.*;

import java.io.Serializable;

/**
 * This class is configuration bean for MemoryWorkspace.
 * It allows you to specify workspace parameters, and will define workspace behaviour.
 *
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

    /**
     * This variable specifies amount of memory allocated for this workspace during initialization
     */
    protected long initialSize;

    /**
     * This variable specifies minimal workspace size
     */
    protected long minSize;

    /**
     * This variable specifies maximal workspace size
     */
    protected long maxSize;

    /**
     * For workspaces with learnable size, this variable defines how many cycles will be spent during learning phase
     */
    protected int cyclesBeforeInitialization;

    /**
     * If OVERALLOCATION policy is set, memory will be overallocated in addition to initialSize of learned size
     */
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
