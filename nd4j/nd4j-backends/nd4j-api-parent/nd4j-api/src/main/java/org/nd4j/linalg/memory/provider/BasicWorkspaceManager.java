/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.memory.provider;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MemoryWorkspaceManager;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.memory.pointers.PointersPair;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.memory.abstracts.Nd4jWorkspace;
import org.nd4j.linalg.primitives.SynchronizedObject;
import org.nd4j.util.StringUtils;

import java.lang.ref.ReferenceQueue;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;


/**
 * Workspace manager implementation. Please note, this class is supposed to be used via Nd4j.getWorkspaceManager(), to provide consistency between different threads within given JVM process
 * @author raver119@gmail.com
 */
@Slf4j
public abstract class BasicWorkspaceManager implements MemoryWorkspaceManager {

    protected AtomicLong counter = new AtomicLong();
    protected WorkspaceConfiguration defaultConfiguration;
    protected ThreadLocal<Map<String, MemoryWorkspace>> backingMap = new ThreadLocal<>();
    private ReferenceQueue<MemoryWorkspace> queue;
    private WorkspaceDeallocatorThread thread;
    private Map<String, Nd4jWorkspace.GarbageWorkspaceReference> referenceMap = new ConcurrentHashMap<>();

    // default mode is DISABLED, as in: production mode
    protected SynchronizedObject<DebugMode> debugMode = new SynchronizedObject<>(DebugMode.DISABLED);

    public BasicWorkspaceManager() {
        this(WorkspaceConfiguration.builder().initialSize(0).maxSize(0).overallocationLimit(0.3)
                        .policyAllocation(AllocationPolicy.OVERALLOCATE).policyLearning(LearningPolicy.FIRST_LOOP)
                        .policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.EXTERNAL).build());
    }

    public BasicWorkspaceManager(@NonNull WorkspaceConfiguration defaultConfiguration) {
        this.defaultConfiguration = defaultConfiguration;
        this.queue = new ReferenceQueue<>();

        thread = new WorkspaceDeallocatorThread(this.queue);
        thread.start();
    }

    /**
     * Returns globally unique ID
     *
     * @return
     */
    @Override
    public String getUUID() {
        return "Workspace_" + String.valueOf(counter.incrementAndGet());
    }

    /**
     * This method allows to specify "Default" configuration, that will be used in signatures which do not have WorkspaceConfiguration argument
     * @param configuration
     */
    @Override
    public void setDefaultWorkspaceConfiguration(@NonNull WorkspaceConfiguration configuration) {
        this.defaultConfiguration = configuration;
    }

    /**
     * This method will return workspace with default configuration and default id.
     * @return
     */
    @Override
    public MemoryWorkspace getWorkspaceForCurrentThread() {
        return getWorkspaceForCurrentThread(MemoryWorkspace.DEFAULT_ID);
    }

    @Override
    public MemoryWorkspace getWorkspaceForCurrentThread(@NonNull String id) {
        return getWorkspaceForCurrentThread(defaultConfiguration, id);
    }

    @Override
    public DebugMode getDebugMode() {
        return debugMode.get();
    }

    @Override
    public void setDebugMode(DebugMode mode) {
        if (mode == null)
            mode = DebugMode.DISABLED;

        debugMode.set(mode);
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
        Nd4jWorkspace.GarbageWorkspaceReference reference =
                        new Nd4jWorkspace.GarbageWorkspaceReference(workspace, queue);
        referenceMap.put(reference.getKey(), reference);
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

    /**
     * This method destroys given workspace
     *
     * @param workspace
     */
    @Override
    public void destroyWorkspace(MemoryWorkspace workspace) {
        if (workspace == null || workspace instanceof DummyWorkspace)
            return;

        //workspace.destroyWorkspace();
        backingMap.get().remove(workspace.getId());
    }

    /**
     * This method destroy default workspace, if any
     */
    @Override
    public void destroyWorkspace() {
        ensureThreadExistense();

        MemoryWorkspace workspace = backingMap.get().get(MemoryWorkspace.DEFAULT_ID);
        //if (workspace != null)
        //workspace.destroyWorkspace();

        backingMap.get().remove(MemoryWorkspace.DEFAULT_ID);
    }

    /**
     * This method destroys all workspaces allocated in current thread
     */
    @Override
    public void destroyAllWorkspacesForCurrentThread() {
        ensureThreadExistense();

        List<MemoryWorkspace> workspaces = new ArrayList<>();
        workspaces.addAll(backingMap.get().values());

        for (MemoryWorkspace workspace : workspaces) {
            destroyWorkspace(workspace);
        }

        System.gc();
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
    public MemoryWorkspace getAndActivateWorkspace(@NonNull WorkspaceConfiguration configuration, @NonNull String id) {
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


    @Override
    public boolean checkIfWorkspaceExistsAndActive(@NonNull String id) {
        boolean exists = checkIfWorkspaceExists(id);
        if (!exists)
            return false;

        return backingMap.get().get(id).isScopeActive();
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
            return new DummyWorkspace();
        else {
            Nd4j.getMemoryManager().setCurrentWorkspace(null);
            return workspace.tagOutOfScopeUse();
        }
    }


    @Deprecated // For test use within the github.com/deeplearning4j/deeplearning4j repo only.
    public static final String WorkspaceDeallocatorThreadName = "Workspace deallocator thread";

    protected class WorkspaceDeallocatorThread extends Thread implements Runnable {
        private final ReferenceQueue<MemoryWorkspace> queue;

        protected WorkspaceDeallocatorThread(ReferenceQueue<MemoryWorkspace> queue) {
            this.queue = queue;
            this.setDaemon(true);
            this.setName(WorkspaceDeallocatorThreadName);
        }

        @Override
        public void run() {
            while (true) {
                try {
                    Nd4jWorkspace.GarbageWorkspaceReference reference =
                                    (Nd4jWorkspace.GarbageWorkspaceReference) queue.remove();
                    if (reference != null) {
                        //                      log.info("Releasing reference for Workspace [{}]", reference.getId());
                        PointersPair pair = reference.getPointersPair();
                        // purging workspace planes
                        if (pair != null) {
                            if (pair.getDevicePointer() != null) {
                                //log.info("Deallocating device...");
                                Nd4j.getMemoryManager().release(pair.getDevicePointer(), MemoryKind.DEVICE);
                            }


                            if (pair.getHostPointer() != null) {
                                //                                log.info("Deallocating host...");
                                Nd4j.getMemoryManager().release(pair.getHostPointer(), MemoryKind.HOST);
                            }
                        }

                        // purging all spilled pointers
                        for (PointersPair pair2 : reference.getExternalPointers()) {
                            if (pair2 != null) {
                                if (pair2.getHostPointer() != null)
                                    Nd4j.getMemoryManager().release(pair2.getHostPointer(), MemoryKind.HOST);

                                if (pair2.getDevicePointer() != null)
                                    Nd4j.getMemoryManager().release(pair2.getDevicePointer(), MemoryKind.DEVICE);
                            }
                        }

                        // purging all pinned pointers
                        while ((pair = reference.getPinnedPointers().poll()) != null) {
                            if (pair.getHostPointer() != null)
                                Nd4j.getMemoryManager().release(pair.getHostPointer(), MemoryKind.HOST);

                            if (pair.getDevicePointer() != null)
                                Nd4j.getMemoryManager().release(pair.getDevicePointer(), MemoryKind.DEVICE);
                        }

                        referenceMap.remove(reference.getKey());
                    }
                } catch (InterruptedException e) {
                    return; /* terminate thread when being interrupted */
                } catch (Exception e) {
                    //
                }
            }
        }
    }

    /**
     * This method prints out basic statistics for workspaces allocated in current thread
     */
    public synchronized void printAllocationStatisticsForCurrentThread() {
        ensureThreadExistense();
        Map<String, MemoryWorkspace> map = backingMap.get();
        log.info("Workspace statistics: ---------------------------------");
        log.info("Number of workspaces in current thread: {}", map.size());
        log.info("Workspace name: Allocated / external (spilled) / external (pinned)");
        for (String key : map.keySet()) {
            long current = ((Nd4jWorkspace) map.get(key)).getCurrentSize();
            long spilled = ((Nd4jWorkspace) map.get(key)).getSpilledSize();
            long pinned = ((Nd4jWorkspace) map.get(key)).getPinnedSize();
            log.info(String.format("%-26s %8s / %8s / %8s (%11d / %11d / %11d)", (key + ":"),
                    StringUtils.TraditionalBinaryPrefix.long2String(current, "", 2),
                    StringUtils.TraditionalBinaryPrefix.long2String(spilled, "", 2),
                    StringUtils.TraditionalBinaryPrefix.long2String(pinned, "", 2),
                    current, spilled, pinned));
        }
    }


    @Override
    public List<String> getAllWorkspacesIdsForCurrentThread() {
        ensureThreadExistense();
        return new ArrayList<>(backingMap.get().keySet());
    }

    @Override
    public List<MemoryWorkspace> getAllWorkspacesForCurrentThread() {
        ensureThreadExistense();
        return new ArrayList<>(backingMap.get().values());
    }

    @Override
    public boolean anyWorkspaceActiveForCurrentThread(){
        ensureThreadExistense();
        boolean anyActive = false;
        for(MemoryWorkspace ws : backingMap.get().values()){
            if(ws.isScopeActive()){
                anyActive = true;
                break;
            }
        }
        return anyActive;
    }
}
