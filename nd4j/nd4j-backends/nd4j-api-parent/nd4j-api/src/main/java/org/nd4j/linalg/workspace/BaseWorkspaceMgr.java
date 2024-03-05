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

package org.nd4j.linalg.workspace;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MemoryWorkspaceManager;
import org.nd4j.linalg.api.memory.WorkspaceUseMetaData;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEvent;
import org.nd4j.linalg.profiler.data.array.event.NDArrayEventType;
import org.nd4j.linalg.profiler.data.array.event.NDArrayMetaData;

import java.util.*;
import java.util.concurrent.ConcurrentSkipListSet;

import static org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner.allOpenWorkspaces;

/**
 * A {@link WorkspaceMgr} for use with {@link INDArray} instances.<br>
 * This class handles the creation and initializiation of workspaces, and provides methods for validating that arrays
 * are in the correct workspace, and leveraging arrays to the correct workspace.<br>
 * <br>
 * <b>Usage:</b><br>
 * {@code
 * WorkspaceConfiguration conf = WorkspaceConfiguration.builder()
 *     .initialSize(10 * 1024L * 1024L)  //10MB initial workspace size
 *     .overallocationLimit(3.0)         //Allocate 3x initialSize as workspace grows
 *     .policyAllocation(AllocationPolicy.OVERALLOCATE)  //Allocate over and above what was initially requested
 *     .policyLearning(LearningPolicy.FIRST_LOOP)        //Learn overallocation after first iteration
 *     .policyMirroring(MirroringPolicy.FULL)            //Preallocate workspace for use on both host (CPU) and device (GPU)
 *     .policySpill(SpillPolicy.REALLOCATE)              //Reallocate workspaces when they are full. This means data will be copied
 *     .build();                                             // from workspace to system memory (and back) when necessary
 *                                                  //Use SPILL_EXTERNAL if you want workspaces to spill directly to disk when full
 *                                                  //Use SPILL_RESET for behavior like SPILL_EXTERNAL, but the workspace will be
 *                                                  // automatically reset to 0 if/when it is closed
 *                                                  //Use NO_SPILL to disable spilling. Once the workspace is full, any further
 *                                                  // attempt to allocate memory will result in an exception
 *                                                  //Use EXTERNAL if you want workspaces to spill directly to disk when full
 *                                                  //Use RESET for behavior like EXTERNAL, but the workspace will be
 *                                                  // automatically reset to 0 if/when it is closed
 *                                                  //Use NONE to disable spilling. Once the workspace is full, any further
 *                                                  // attempt to allocate memory will result in an exception
 *                                                  //Use ALWAYS if you want workspaces to always spill to disk when full
 *
 * @param <T> Enum type for the array type
 */
@Slf4j
public abstract class BaseWorkspaceMgr<T extends Enum<T>> implements WorkspaceMgr<T> {
    private static final boolean DISABLE_LEVERAGE = false;  //Mainly for debugging/optimization purposes

    protected  Set<T> scopeOutOfWs;
    protected  Set<T> keepTypesOpen = new ConcurrentSkipListSet<>();
    protected  Map<T, WorkspaceConfiguration> configMap;
    protected  Map<T, String> workspaceNames;


    @Override
    public void keepOpen(T... types) {
        if(types != null)
            keepTypesOpen.addAll(Arrays.asList(types));
        for(T workspaceType : types) {
            if(configMap.containsKey(workspaceType)) {
                notifyScopeEntered(workspaceType);
            }
        }
    }

    @Override
    public void removeKeepOpen(T... types) {
        keepTypesOpen.removeAll(Arrays.asList(types));
    }



    @Override
    public void recordWorkspaceClose(MemoryWorkspace workspace, T type) {
        recordWorkspaceEvent(WorkspaceUseMetaData.EventTypes.CLOSE,workspace, type);
    }

    @Override
    public void recordWorkspaceOpen(MemoryWorkspace workspace, T arrayType) {
        recordWorkspaceEvent(WorkspaceUseMetaData.EventTypes.ENTER,workspace, arrayType);
    }


    @Override
    public void recordWorkspaceBorrow(MemoryWorkspace workspace, T type) {
        recordWorkspaceEvent(WorkspaceUseMetaData.EventTypes.BORROW,workspace, type);
    }


    @Override
    public void recordWorkspaceSet(MemoryWorkspace workspace, T type) {
        recordWorkspaceEvent(WorkspaceUseMetaData.EventTypes.SET_CURRENT,workspace, type);
    }


    @Override
    public void closeWorkspace(T... types) {
        for(T type : types) {
            if(configMap.containsKey(type)) {
                String workspaceName = getWorkspaceName(type);
                Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceName).close();
            }
        }
    }

    protected void recordWorkspaceEvent(WorkspaceUseMetaData.EventTypes eventType, MemoryWorkspace workspace, T arrayType) {
        if(workspace == null)
            return;
        WorkspaceUseMetaData workspaceUseMetaData = WorkspaceUseMetaData.builder()
                .stackTrace(Thread.currentThread().getStackTrace())
                .workspaceActivateAtTimeOfEvent(workspace.isScopeActive())
                .generation(workspace.getGenerationId())
                .uniqueId(workspace.getUniqueId())
                .workspaceSize(workspace.getCurrentSize())
                .lastCycleAllocations(workspace.getLastCycleAllocations())
                .eventTime(System.currentTimeMillis())
                .workspaceName(workspace.getId())
                .eventType(eventType)
                .associatedEnum(arrayType)
                .threadName(Thread.currentThread().getName())
                .build();
        Nd4j.getExecutioner().getNd4jEventLog().recordWorkspaceEvent(workspaceUseMetaData);

    }


    protected BaseWorkspaceMgr(Set<T> scopeOutOfWs, Map<T, WorkspaceConfiguration> configMap,
                               Map<T, String> workspaceNames) {
        this.scopeOutOfWs = scopeOutOfWs;
        this.configMap = configMap;
        this.workspaceNames = workspaceNames;
    }

    protected BaseWorkspaceMgr() {
        scopeOutOfWs = new HashSet<>();
        configMap = new HashMap<>();
        workspaceNames = new HashMap<>();
    }

    @Override
    public void setConfiguration(@NonNull T arrayType, WorkspaceConfiguration configuration) {
        configMap.put(arrayType, configuration);
    }

    @Override
    public WorkspaceConfiguration getConfiguration(@NonNull T arrayType) {
        return configMap.get(arrayType);
    }

    @Override
    public void setScopedOutFor(@NonNull T arrayType) {
        scopeOutOfWs.add(arrayType);
    }

    @Override
    public boolean isScopedOut(@NonNull T arrayType) {
        return scopeOutOfWs.contains(arrayType);
    }

    @Override
    public boolean hasConfiguration(@NonNull T arrayType) {
        return scopeOutOfWs.contains(arrayType) || workspaceNames.containsKey(arrayType);
    }

    @Override
    public MemoryWorkspace notifyScopeEntered(@NonNull T arrayType) {
        validateConfig(arrayType);
        if(isScopedOut(arrayType)) {
            recordWorkspaceOpen(Nd4j.getWorkspaceManager().scopeOutOfWorkspaces(), arrayType);
            return Nd4j.getWorkspaceManager().scopeOutOfWorkspaces();
        } else {
            MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                    getConfiguration(arrayType), getWorkspaceName(arrayType));
            ws.setAssociatedEnumType(arrayType);
            ws.setWorkspaceMgr(this);
            recordWorkspaceOpen(ws, arrayType);

            return ws.notifyScopeEntered();
        }
    }

    @Override
    public WorkspacesCloseable notifyScopeEntered(@NonNull T... arrayTypes) {
        MemoryWorkspace[] ws = new MemoryWorkspace[arrayTypes.length];
        for(int i = 0; i < arrayTypes.length; i++) {
            ws[i] = notifyScopeEntered(arrayTypes[i]);
        }
        return new WorkspacesCloseable(ws);
    }

    @Override
    public MemoryWorkspace notifyScopeBorrowed(@NonNull T arrayType) {
        validateConfig(arrayType);
        enforceExistsAndActive(arrayType);

        if(scopeOutOfWs.contains(arrayType)) {
            recordWorkspaceBorrow(Nd4j.getWorkspaceManager().scopeOutOfWorkspaces(), arrayType);
            return Nd4j.getWorkspaceManager().scopeOutOfWorkspaces();
        } else {
            MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                    getConfiguration(arrayType), getWorkspaceName(arrayType));
            ws.setWorkspaceMgr(this);
            recordWorkspaceBorrow(ws, arrayType);

            return ws.notifyScopeBorrowed();
        }
    }

    @Override
    public void setWorkspaceName(@NonNull T arrayType, @NonNull String name) {
        workspaceNames.put(arrayType, name);
    }

    @Override
    public String getWorkspaceName(@NonNull T arrayType) {
        return workspaceNames.get(arrayType);
    }

    @Override
    public void setWorkspace(@NonNull T forEnum, @NonNull String wsName, @NonNull WorkspaceConfiguration configuration) {
        if(scopeOutOfWs.contains(forEnum)) {
            scopeOutOfWs.remove(forEnum);
        }
        setWorkspaceName(forEnum, wsName);
        setConfiguration(forEnum, configuration);
    }

    @Override
    public boolean isWorkspaceOpen(@NonNull T arrayType) {
        validateConfig(arrayType);
        if(!scopeOutOfWs.contains(arrayType)) {
            return Nd4j.getWorkspaceManager().checkIfWorkspaceExistsAndActive(getWorkspaceName(arrayType));
        }
        return true;
    }

    @Override
    public void assertOpen(T arrayType, String msg) throws ND4JWorkspaceException {
        if(!scopeOutOfWs.contains(arrayType) && !isWorkspaceOpen(arrayType)) {
            throw new ND4JWorkspaceException("Assertion failed: expected workspace for array type " + arrayType
                    + " to be open: " + msg);
        }
    }

    @Override
    public void assertNotOpen(@NonNull T arrayType, @NonNull String msg) {
        if(!scopeOutOfWs.contains(arrayType) && isWorkspaceOpen(arrayType)){
            throw new ND4JWorkspaceException("Assertion failed: expected workspace for array type " + arrayType
                    + " to not be open: " + msg);
        }
    }

    @Override
    public void setCurrentWorkspace(T arrayType) {
        validateConfig(arrayType);
        if(scopeOutOfWs.contains(arrayType)) {
            Nd4j.getMemoryManager().setCurrentWorkspace(null);
        } else {
            String workspaceName = getWorkspaceName(arrayType);
            MemoryWorkspaceManager workspaceManager = Nd4j.getWorkspaceManager();
            MemoryWorkspace workspaceForCurrentThread = workspaceManager.getWorkspaceForCurrentThread(workspaceName);
            workspaceForCurrentThread.setAssociatedEnumType(arrayType);
            recordWorkspaceSet(workspaceForCurrentThread, arrayType);
            if(workspaceForCurrentThread != null) {
                Nd4j.getMemoryManager().setCurrentWorkspace(workspaceForCurrentThread);
            } else {
                throw new IllegalArgumentException("Workspace for array type " + arrayType + " not found.");
            }
        }
    }

    @Override
    public void assertCurrentWorkspace(@NonNull T arrayType, String msg) {
        validateConfig(arrayType);
        MemoryWorkspace curr = Nd4j.getMemoryManager().getCurrentWorkspace();
        if(!scopeOutOfWs.contains(arrayType) && (curr == null || !getWorkspaceName(arrayType).equals(curr.getId()))) {
            throw new ND4JWorkspaceException("Assertion failed: expected current workspace to be \"" + getWorkspaceName(arrayType)
                    + "\" (for array type " + arrayType + ") - actual current workspace is " + (curr == null ? null : curr.getId())
                    + (msg == null ? "" : ": " + msg));
        };
    }

    @Override
    public INDArray leverageTo(@NonNull T arrayType, @NonNull INDArray array) {
        if(array == null || !array.isAttached()) {
            if(Nd4j.getEnvironment().isLogNDArrayEvents()) {
                Nd4j.getExecutioner().getNd4jEventLog().addToNDArrayLog(array.getId(),
                        NDArrayEvent.builder()
                                .stackTrace(Thread.currentThread().getStackTrace())
                                .dataAtEvent(NDArrayMetaData.from(array))
                                .ndArrayEventType(NDArrayEventType.ARRAY_WORKSPACE_LEVERAGE)
                                .build());
            }

            return array;
        }

        validateConfig(arrayType);
        enforceExistsAndActive(arrayType);

        if(!DISABLE_LEVERAGE) {
            if(scopeOutOfWs.contains(arrayType)) {
                return array.detach();
            }
            return array.leverageTo(getWorkspaceName(arrayType), true);
        } else {
            if(array.isAttached()) {
                if(!array.data().getParentWorkspace().getId().equals(getWorkspaceName(arrayType))) {
                    throw new IllegalStateException("Array of type " + arrayType + " is leveraged from " + array.data().getParentWorkspace().getId()
                            + " to " + getWorkspaceName(arrayType) + " but WorkspaceMgn.leverageTo() is currently disabled");
                }
            }
            return array;
        }
    }

    @Override
    public INDArray validateArrayLocation(@NonNull T arrayType, @NonNull INDArray array, boolean migrateIfInvalid, boolean exceptionIfDetached) {
        validateConfig(arrayType);

        if(scopeOutOfWs.contains(arrayType)) {
            //Array is supposed to be detached (no workspace)
            boolean ok = !array.isAttached();
            if(!ok) {
                if(migrateIfInvalid) {
                    log.trace("Migrating array of type " + arrayType + " to workspace " + getWorkspaceName(arrayType));
                    return leverageTo(arrayType, array);
                } else {
                    throw new ND4JWorkspaceException("Array workspace validation failed: Array of type " + arrayType
                            + " should be detached (no workspace) but is in workspace: " + array.data().getParentWorkspace().getId());
                }
            } else {
                //Detached array, as expected
                return array;
            }
        }

        //At this point: we expect the array to be in a workspace
        String wsNameExpected = getWorkspaceName(arrayType);
        if(!array.isAttached()) {
            if(exceptionIfDetached) {
                throw new ND4JWorkspaceException("Array workspace validation failed: Array of type " + arrayType +
                        " should be in workspace \"" + wsNameExpected + "\" but is detached");
            } else {
                return array;
            }
        }


        String wsNameAct = array.data().getParentWorkspace().getId();
        if(!wsNameExpected.equals(wsNameAct)) {
            if(migrateIfInvalid) {
                return leverageTo(arrayType, array);
            } else {
                throw new ND4JWorkspaceException("Array workspace validation failed: Array of type " + arrayType +
                        " should be in workspace \"" + wsNameExpected + "\" but is in workspace \"" + wsNameAct + "\"");
            }
        }

        //OK - return as-is
        return array;
    }

    @Override
    public INDArray create(@NonNull T arrayType, @NonNull DataType dataType, @NonNull long... shape) {
        enforceExistsAndActive(arrayType);
        return create(arrayType, dataType, shape, Nd4j.order());
    }

    /**
     * This method creates INDArray of specified dataType and shape, and puts it into Workspace, if any.
     * has with respect to the java deallocator service.
     *
     * All crashes now seem to be induced by java side free calls.
     *
     * We need to figure out where these free calls are coming from
     * and why they crash everything. The guess is the way
     * deallocation + workspace closing works. This is the most recent set of changes
     * made.
     *
     * @param arrayType Array type
     * @param dataType  Data type for the created array
     * @param shape     Shape
     * @param order Order of the array
     * @return
     */

    @Override
    public INDArray create(@NonNull T arrayType, @NonNull DataType dataType, @NonNull long[] shape, @NonNull char order) {
        enforceExistsAndActive(arrayType);
        if(keepTypesOpen.contains(arrayType)) {
            return Nd4j.create(dataType, shape, order);

        } else {
            try(MemoryWorkspace ws = notifyScopeBorrowed(arrayType)) {
                return Nd4j.create(dataType, shape, order);
            }
        }

    }

    @Override
    public INDArray createUninitialized(T arrayType, DataType dataType, long... shape) {
        return createUninitialized(arrayType, dataType, shape, Nd4j.order());
    }

    @Override
    public INDArray createUninitialized(@NonNull T arrayType, @NonNull DataType dataType, @NonNull long[] shape, char order) {
        enforceExistsAndActive(arrayType);
        if(keepTypesOpen.contains(arrayType)) {
            String workspaceName = getWorkspaceName(arrayType);
            if(workspaceName != null) {
                MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceName);
                ws.setAssociatedEnumType(arrayType);

                //since we keep scopes open and there is no guarantee the  current array maybe of this workspace
                //we ensure it is with leverage
                INDArray ret = Nd4j.createUninitialized(dataType, shape, order);
                if(ws != ret.getWorkspace()) {
                    return leverageTo(arrayType,ret);
                }
            } else { //scope out of  workspaces when nothing found
                try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                    return Nd4j.createUninitialized(dataType, shape, order);
                }
            }

            return Nd4j.createUninitialized(dataType, shape, order);

        } else {
            try(MemoryWorkspace ws = notifyScopeBorrowed(arrayType)) {
                return Nd4j.createUninitialized(dataType, shape, order);
            }
        }

    }

    @Override
    public INDArray dup(@NonNull T arrayType, @NonNull INDArray toDup, char order) {
        enforceExistsAndActive(arrayType);
        if (keepTypesOpen.contains(arrayType)) {
            String workspaceName = getWorkspaceName(arrayType);
            if(workspaceName != null) {
                MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceName);
                ws.setAssociatedEnumType(arrayType);
                //since we keep scopes open and there is no guarantee the  current array maybe of this workspace
                //we ensure it is with leverage
                if(ws != toDup.getWorkspace()) {
                    return leverageTo(arrayType,toDup.dup(order));
                }
            } else if(workspaceName == null) {
                try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                    return toDup.dup(order);
                }
            }
            else {
                MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceName);
                return leverageTo(arrayType,toDup.dup(order));

            }

            return toDup.dup(order);
        }  else {
            try (MemoryWorkspace ws = notifyScopeBorrowed(arrayType)) {
                return toDup.dup(order);
            }
        }
    }
    @Override
    public INDArray dup(@NonNull T arrayType, @NonNull INDArray toDup) {
        return dup(arrayType, toDup, toDup.ordering());
    }

    @Override
    public INDArray castTo(@NonNull T arrayType, @NonNull DataType dataType, @NonNull INDArray toCast, boolean dupIfCorrectType) {
        if(toCast.dataType() == dataType) {
            if(!dupIfCorrectType) {
                //Check if we can avoid duping... if not in workspace, or already in correct workspace
                if(!toCast.isAttached() || toCast.data().getParentWorkspace().getId().equals(workspaceNames.get(arrayType))) {
                    return toCast;
                }
            }
            return dup(arrayType, toCast);
        } else {
            if(keepTypesOpen.contains(arrayType))
                return toCast.castTo(dataType);
            else {
                try(MemoryWorkspace ws = notifyScopeBorrowed(arrayType)) {
                    return toCast.castTo(dataType);
                }
            }

        }
    }


    private void validateConfig(@NonNull T arrayType) {
        if(scopeOutOfWs.contains(arrayType)) {
            return;
        }

        if(!configMap.containsKey(arrayType)) {
            throw new ND4JWorkspaceException("No workspace configuration has been provided for arrayType: " + arrayType);
        }
        if(!workspaceNames.containsKey(arrayType)) {
            throw new ND4JWorkspaceException("No workspace name has been provided for arrayType: " + arrayType);
        }
    }

    private void enforceExistsAndActive(@NonNull T arrayType) {
        validateConfig(arrayType);
        if(scopeOutOfWs.contains(arrayType)) {
            return;
        }

        if(!Nd4j.getWorkspaceManager().checkIfWorkspaceExistsAndActive(workspaceNames.get(arrayType))) {
            throw new ND4JWorkspaceException("Workspace \"" + workspaceNames.get(arrayType) + "\" for array type " + arrayType
                    + " is not open. Workspaces open: " + allOpenWorkspaces());
        }
    }




}
