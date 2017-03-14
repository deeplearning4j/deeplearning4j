package org.nd4j.linalg.memory.abstracts;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.MemoryKind;
import org.nd4j.linalg.memory.MemoryManager;
import org.nd4j.linalg.memory.MemoryWorkspace;
import org.nd4j.linalg.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.memory.enums.AllocationPolicy;
import org.nd4j.linalg.memory.enums.LearningPolicy;
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
public class Nd4jWorkspace implements AutoCloseable, MemoryWorkspace {
    protected int deviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();


    protected AtomicLong currentSize = new AtomicLong(0);
    protected AtomicLong currentOffset = new AtomicLong(0);

    protected MemoryManager memoryManager;

    protected WorkspaceConfiguration workspaceConfiguration = new WorkspaceConfiguration();

    // TODO: it should be something like our PointersPair
    protected List<PointersPair> externalAllocations = new ArrayList<>();

    // this memory manager implementation will be used to allocate real memory for this workspace



    public Nd4jWorkspace(@NonNull WorkspaceConfiguration configuration) {
        this.workspaceConfiguration = configuration;

        this.memoryManager = Nd4j.getMemoryManager();
    }

    protected void init() {
        //  we want params validation here

        // and actual workspace allocation
        this.currentSize.set(workspaceConfiguration.getInitialSize());
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
}
