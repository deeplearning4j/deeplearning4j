package org.nd4j.linalg.memory.provider;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MemoryWorkspaceManager;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;

import java.lang.ref.ReferenceQueue;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public abstract class BasicWorkspaceManager implements MemoryWorkspaceManager {

    protected WorkspaceConfiguration defaultConfiguration;
    protected ThreadLocal<Map<String, MemoryWorkspace>> backingMap = new ThreadLocal<>();
    private DummyWorkspace dummyWorkspace = new DummyWorkspace();
    private ReferenceQueue<MemoryWorkspace> queue;
    private WorkspaceDeallocatorThread thread;
    private Map<Long, Nd4jWorkspace.GarbageWorkspaceReference> referenceMap = new ConcurrentHashMap<>();

    public BasicWorkspaceManager() {
        this(WorkspaceConfiguration.builder().initialSize(0).maxSize(0).overallocationLimit(0.3).policyAllocation(AllocationPolicy.OVERALLOCATE).policyLearning(LearningPolicy.FIRST_LOOP).policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.EXTERNAL).build());
    }

    public BasicWorkspaceManager(@NonNull WorkspaceConfiguration defaultConfiguration) {
        this.defaultConfiguration = defaultConfiguration;
        this.queue = new ReferenceQueue<>();

        thread = new WorkspaceDeallocatorThread(this.queue);
        thread.start();
    }

    @Override
    public void setDefaultWorkspaceConfiguration(@NonNull WorkspaceConfiguration configuration) {
        this.defaultConfiguration = configuration;
    }

    @Override
    public MemoryWorkspace getWorkspaceForCurrentThread() {
        return getWorkspaceForCurrentThread(MemoryWorkspace.DEFAULT_ID);
    }

    @Override
    public MemoryWorkspace getWorkspaceForCurrentThread(@NonNull String id) {
        return getWorkspaceForCurrentThread(defaultConfiguration, id);
    }

    /*
    @Override
    public MemoryWorkspace getWorkspaceForCurrentThread(@NonNull WorkspaceConfiguration configuration, @NonNull String id) {
        ensureThreadExistense();

        MemoryWorkspace workspace = backingMap.get().get(id);
        if (workspace == null) {
            workspace = new Nd4jWorkspace(configuration, id);
            backingMap.get().put(id, workspace);
        }

        return workspace;
    }
    */

    protected void pickReference(MemoryWorkspace workspace) {
        Nd4jWorkspace.GarbageWorkspaceReference reference = new Nd4jWorkspace.GarbageWorkspaceReference(workspace, queue);
        referenceMap.put(reference.getThreadId(), reference);
    }

    @Override
    public void setWorkspaceForCurrentThread(MemoryWorkspace workspace) {
        setWorkspaceForCurrentThread(workspace, MemoryWorkspace.DEFAULT_ID);
    }

    @Override
    public void setWorkspaceForCurrentThread(@NonNull MemoryWorkspace workspace, @NonNull String id) {
        ensureThreadExistense();

        backingMap.get().put(id, workspace);
    }

    @Override
    public void destroyWorkspace(@NonNull MemoryWorkspace workspace) {
        workspace.destroyWorkspace();
    }

    @Override
    public void destroyWorkspace() {
        ensureThreadExistense();

        MemoryWorkspace workspace = backingMap.get().get(MemoryWorkspace.DEFAULT_ID);
        if (workspace != null)
            workspace.destroyWorkspace();

        backingMap.get().remove(MemoryWorkspace.DEFAULT_ID);
    }

    @Override
    public void destroyAllWorkspacesForCurrentThread() {
        ensureThreadExistense();

        for (String key : backingMap.get().keySet()) {
            backingMap.get().get(key).destroyWorkspace();
        }
    }

    protected void ensureThreadExistense() {
        if (backingMap.get() == null)
            backingMap.set(new HashMap<String, MemoryWorkspace>());
    }

    /**
     * This method gets & activates default workspace
     *
     * @return
     */
    @Override
    public MemoryWorkspace getAndActivateWorkspace() {
        return getWorkspaceForCurrentThread().notifyScopeEntered();
    }

    /**
     * This method gets & activates workspace with a given Id
     *
     * @param id
     * @return
     */
    @Override
    public MemoryWorkspace getAndActivateWorkspace(@NonNull String id) {
        return getWorkspaceForCurrentThread(id).notifyScopeEntered();
    }

    /**
     * This method gets & activates default with a given configuration and Id
     *
     * @param configuration
     * @param id
     * @return
     */
    @Override
    public MemoryWorkspace getAndActivateWorkspace(@NonNull WorkspaceConfiguration configuration,@NonNull String id) {
        return getWorkspaceForCurrentThread(configuration, id).notifyScopeEntered();
    }

    /**
     * This method checks, if Workspace with a given Id was created before this call
     *
     * @param id
     * @return
     */
    @Override
    public boolean checkIfWorkspaceExists(@NonNull String id) {
        ensureThreadExistense();
        return backingMap.get().containsKey(id);
    }


    /**
     * This method temporary opens block out of any workspace scope.
     * <p>
     * PLEASE NOTE: Do not forget to close this block.
     *
     * @return
     */
    @Override
    public MemoryWorkspace scopeOutOfWorkspaces() {
        MemoryWorkspace workspace = Nd4j.getMemoryManager().getCurrentWorkspace();
        if (workspace == null)
            return dummyWorkspace;
        else {
            Nd4j.getMemoryManager().setCurrentWorkspace(null);
            return workspace.tagOutOfScopeUse();
        }
    }


    protected class WorkspaceDeallocatorThread extends Thread implements Runnable {
        private final ReferenceQueue<MemoryWorkspace> queue;

        protected WorkspaceDeallocatorThread(ReferenceQueue<MemoryWorkspace> queue) {
            this.queue = queue;
            this.setDaemon(true);
            this.setName("Workspace deallocator thread");
        }

        @Override
        public void run() {
            while (true) {
                try {
                    Nd4jWorkspace.GarbageWorkspaceReference reference = (Nd4jWorkspace.GarbageWorkspaceReference) queue.remove();
                    if (reference != null) {
                        //log.info("Releasing reference for Workspace [{}]", reference.getId());
                        PagedPointer ptrDevice = reference.getPointerDevice();
                        if (ptrDevice != null)
                            Nd4j.getMemoryManager().release(ptrDevice, MemoryKind.DEVICE);

                        PagedPointer ptrHost = reference.getPointerHost();
                        if (ptrHost != null) {
                            referenceMap.remove(ptrHost.address());
                            Nd4j.getMemoryManager().release(ptrHost, MemoryKind.HOST);
                        }
                    }
                } catch (Exception e) {
                    //
                }
            }
        }
    }
}
