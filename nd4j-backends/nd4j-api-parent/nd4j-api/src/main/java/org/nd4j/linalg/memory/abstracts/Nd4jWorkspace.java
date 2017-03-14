package org.nd4j.linalg.memory.abstracts;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.MemoryKind;
import org.nd4j.linalg.memory.MemoryManager;
import org.nd4j.linalg.memory.enums.AllocationPolicy;
import org.nd4j.linalg.memory.enums.MirroringPolicy;
import org.nd4j.linalg.memory.enums.SpillPolicy;
import org.nd4j.linalg.memory.pointers.PointersPair;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class Nd4jWorkspace implements AutoCloseable {
    protected int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

    protected AtomicLong initialSize = new AtomicLong(0);
    protected AtomicLong maxSize = new AtomicLong(0);
    protected AtomicLong currentSize = new AtomicLong(0);
    protected AtomicLong currentOffset = new AtomicLong(0);

    // TODO: it should be something like our PointersPair
    protected List<PointersPair> externalAllocations = new ArrayList<>();

    // this memory manager implementation will be used to allocate real memory for this workspace
    protected MemoryManager memoryManager;

    protected final AllocationPolicy policyAllocation;
    protected final SpillPolicy policySpill;
    protected final MirroringPolicy policyMirroring;

    protected double overallocationLimit = 0.0;


    protected Nd4jWorkspace(@NonNull AllocationPolicy allocationPolicy, @NonNull SpillPolicy spillPolicy, @NonNull MirroringPolicy mirroringPolicy) {
        this.policyAllocation = allocationPolicy;
        this.policySpill = spillPolicy;
        this.policyMirroring = mirroringPolicy;

        this.memoryManager = Nd4j.getMemoryManager();
    }

    protected void init() {
        //  we want params validation here

        // and actual workspace allocation
        this.currentSize.set(this.initialSize.get());
    }


    public Pointer alloc(long requiredMemory, MemoryKind kind) {
        /*
            just two options here:
            1) reqMem + currentOffset < totalSize, we just return pointer + offset
            2) go for either realloc or external allocation
         */

        return null;
    }

    public void free(Pointer pointer) {
        // no-op for main page(s), purge for external stuff
    }


    @Override
    public void close() throws Exception {
        // TODO: to be implemented
        /*
            Basically all we want here, is:
            1) memset primary page(s)
            2) purge external allocations
         */
    }


    public static class Builder {
        private Nd4jWorkspace workspace;

        public Builder(@NonNull AllocationPolicy allocationPolicy, @NonNull SpillPolicy spillPolicy, @NonNull MirroringPolicy mirroringPolicy) {
            workspace = new Nd4jWorkspace(allocationPolicy, spillPolicy, mirroringPolicy);
        }

        public Builder setOverallocationLimit(double limit) {
            workspace.overallocationLimit = limit;
            return this;
        }

        public Builder setInitialPageSize(long size) {
            workspace.initialSize.set(size);
            return this;
        }

        public Builder setMaximalPageSize(long size) {
            workspace.maxSize.set(size);
            return this;
        }

        public Builder setTargetDeviceId(int deviceId) {
            if (deviceId < 0)
                throw new ND4JIllegalStateException("Target deviceId should be positive value");

            workspace.deviceId = deviceId;
            return this;
        }


        public Nd4jWorkspace build() {
            workspace.init();

            return workspace;
        }
    }
}
