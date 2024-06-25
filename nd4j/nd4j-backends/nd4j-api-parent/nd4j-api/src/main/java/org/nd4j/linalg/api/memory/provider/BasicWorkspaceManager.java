/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.memory.provider;

import com.jakewharton.byteunits.BinaryByteUnit;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MemoryWorkspaceManager;
import org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.abstracts.DummyWorkspace;
import org.nd4j.common.primitives.SynchronizedObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;


@Slf4j
public abstract class BasicWorkspaceManager implements MemoryWorkspaceManager {

    protected AtomicLong counter = new AtomicLong();
    protected WorkspaceConfiguration defaultConfiguration;
    protected ThreadLocal<Map<String, MemoryWorkspace>> backingMap =ThreadLocal.withInitial(ConcurrentHashMap::new);

    // default mode is DISABLED, as in: production mode
    protected SynchronizedObject<DebugMode> debugMode = new SynchronizedObject<>(DebugMode.DISABLED);

    public BasicWorkspaceManager() {
        this(WorkspaceConfiguration.builder().initialSize(0).maxSize(0).overallocationLimit(0.3)
                .policyAllocation(AllocationPolicy.OVERALLOCATE).policyLearning(LearningPolicy.FIRST_LOOP)
                .policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.EXTERNAL).build());
    }

    public BasicWorkspaceManager(@NonNull WorkspaceConfiguration defaultConfiguration) {
        this.defaultConfiguration = defaultConfiguration;
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



    protected abstract void pickReference(MemoryWorkspace workspace);




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

        workspace.destroyWorkspace(true);
        backingMap.get().remove(workspace.getId());
    }

    /**
     * This method destroy default workspace, if any
     */
    @Override
    public void destroyWorkspace() {
        ensureThreadExistense();

        MemoryWorkspace workspace = backingMap.get().get(MemoryWorkspace.DEFAULT_ID);


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

        Nd4j.getMemoryManager().invokeGc();
    }

    protected void ensureThreadExistense() {
        if (backingMap.get() == null)
            backingMap.set(new HashMap<>());
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
            long current = map.get(key).getCurrentSize();
            if(map.get(key) instanceof Nd4jWorkspace) {
                long spilled = ((Nd4jWorkspace) map.get(key)).getSpilledSize();
                long pinned = ((Nd4jWorkspace) map.get(key)).getPinnedSize();
                log.info(String.format("%-26s %8s / %8s / %8s (%11d / %11d / %11d)", (key + ":"),
                        BinaryByteUnit.format(current, "#.00"),
                        BinaryByteUnit.format(spilled, "#.00"),
                        BinaryByteUnit.format(pinned, "#.00"),
                        current, spilled, pinned));
            }

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
        for(MemoryWorkspace ws : backingMap.get().values()) {
            if(ws.isScopeActive()) {
                anyActive = true;
                break;
            }
        }
        return anyActive;
    }
}
