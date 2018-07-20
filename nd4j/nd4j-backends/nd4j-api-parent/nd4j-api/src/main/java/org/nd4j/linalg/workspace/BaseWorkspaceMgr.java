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

package org.nd4j.linalg.workspace;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * A standard/baseline {@link WorkspaceMgr} implementation
 *
 * @param <T> Array type
 * @author Alex Black
 */
@Slf4j
public abstract class BaseWorkspaceMgr<T extends Enum<T>> implements WorkspaceMgr<T> {
    private static final boolean DISABLE_LEVERAGE = false;  //Mainly for debugging/optimization purposes

    protected final Set<T> scopeOutOfWs;
    protected final Map<T, WorkspaceConfiguration> configMap;
    protected final Map<T, String> workspaceNames;

    protected BaseWorkspaceMgr(Set<T> scopeOutOfWs, Map<T, WorkspaceConfiguration> configMap,
                               Map<T, String> workspaceNames){
        this.scopeOutOfWs = scopeOutOfWs;
        this.configMap = configMap;
        this.workspaceNames = workspaceNames;
    }

    protected BaseWorkspaceMgr(){
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
        configMap.remove(arrayType);
        workspaceNames.remove(arrayType);
    }

    @Override
    public boolean isScopedOut(@NonNull T arrayType) {
        return scopeOutOfWs.contains(arrayType);
    }

    @Override
    public boolean hasConfiguration(@NonNull T arrayType){
        return scopeOutOfWs.contains(arrayType) || workspaceNames.containsKey(arrayType);
    }

    @Override
    public MemoryWorkspace notifyScopeEntered(@NonNull T arrayType) {
        validateConfig(arrayType);

        if(isScopedOut(arrayType)){
            return Nd4j.getWorkspaceManager().scopeOutOfWorkspaces();
        } else {
            MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                    getConfiguration(arrayType), getWorkspaceName(arrayType));
            return ws.notifyScopeEntered();
        }
    }

    @Override
    public WorkspacesCloseable notifyScopeEntered(@NonNull T... arrayTypes) {
        MemoryWorkspace[] ws = new MemoryWorkspace[arrayTypes.length];
        for(int i=0; i<arrayTypes.length; i++ ){
            ws[i] = notifyScopeEntered(arrayTypes[i]);
        }
        return new WorkspacesCloseable(ws);
    }

    @Override
    public MemoryWorkspace notifyScopeBorrowed(@NonNull T arrayType) {
        validateConfig(arrayType);
        enforceExistsAndActive(arrayType);

        if(scopeOutOfWs.contains(arrayType)){
            return Nd4j.getWorkspaceManager().scopeOutOfWorkspaces();
        } else {
            MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                    getConfiguration(arrayType), getWorkspaceName(arrayType));
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
        if(scopeOutOfWs.contains(forEnum)){
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
        if(!scopeOutOfWs.contains(arrayType) && !isWorkspaceOpen(arrayType)){
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
    public void assertCurrentWorkspace(@NonNull T arrayType, String msg) {
        validateConfig(arrayType);
        MemoryWorkspace curr = Nd4j.getMemoryManager().getCurrentWorkspace();
        if(!scopeOutOfWs.contains(arrayType) && (curr == null || !getWorkspaceName(arrayType).equals(curr.getId()))){
            throw new ND4JWorkspaceException("Assertion failed: expected current workspace to be \"" + getWorkspaceName(arrayType)
                    + "\" (for array type " + arrayType + ") - actual current workspace is " + (curr == null ? null : curr.getId())
                    + (msg == null ? "" : ": " + msg));
        };
    }

    @Override
    public INDArray leverageTo(@NonNull T arrayType, @NonNull INDArray array) {
        if(array == null || !array.isAttached()){
            return array;
        }
        validateConfig(arrayType);
        enforceExistsAndActive(arrayType);

        if(!DISABLE_LEVERAGE){
            if(scopeOutOfWs.contains(arrayType)){
                return array.detach();
            }
            return array.leverageTo(getWorkspaceName(arrayType), true);
        } else {
            if(array.isAttached()){
                if(!array.data().getParentWorkspace().getId().equals(getWorkspaceName(arrayType))){
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

        if(scopeOutOfWs.contains(arrayType)){
            //Array is supposed to be detached (no workspace)
            boolean ok = !array.isAttached();
            if(!ok){
                if(migrateIfInvalid){
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
        if(!array.isAttached()){
            if(exceptionIfDetached) {
                throw new ND4JWorkspaceException("Array workspace validation failed: Array of type " + arrayType +
                        " should be in workspace \"" + wsNameExpected + "\" but is detached");
            } else {
                return array;
            }
        }


        String wsNameAct = array.data().getParentWorkspace().getId();
        if(!wsNameExpected.equals(wsNameAct)){
            if(migrateIfInvalid){
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
    public INDArray create(@NonNull T arrayType, @NonNull int[] shape) {
        enforceExistsAndActive(arrayType);
        return create(arrayType, shape, Nd4j.order());
    }

    @Override
    public INDArray create(@NonNull T arrayType, @NonNull long... shape) {
        enforceExistsAndActive(arrayType);
        return create(arrayType, shape, Nd4j.order());
    }

    @Override
    public INDArray create(@NonNull T arrayType, @NonNull int[] shape, @NonNull char order) {
        enforceExistsAndActive(arrayType);
        try(MemoryWorkspace ws = notifyScopeBorrowed(arrayType)){
            return Nd4j.create(shape, order);
        }
    }

    @Override
    public INDArray create(@NonNull T arrayType, @NonNull long[] shape, @NonNull char order) {
        enforceExistsAndActive(arrayType);
        try(MemoryWorkspace ws = notifyScopeBorrowed(arrayType)){
            return Nd4j.create(shape, order);
        }
    }

    @Override
    public INDArray createUninitialized(@NonNull T arrayType, @NonNull int[] shape) {
        return createUninitialized(arrayType, shape, Nd4j.order());
    }

    @Override
    public INDArray createUninitialized(@NonNull T arrayType, @NonNull long... shape) {
        return createUninitialized(arrayType, shape, Nd4j.order());
    }

    @Override
    public INDArray createUninitialized(@NonNull T arrayType, @NonNull int[] shape, char order) {
        enforceExistsAndActive(arrayType);
        try(MemoryWorkspace ws = notifyScopeBorrowed(arrayType)){
            return Nd4j.createUninitialized(shape, order);
        }
    }

    @Override
    public INDArray createUninitialized(@NonNull T arrayType, @NonNull long[] shape, char order) {
        enforceExistsAndActive(arrayType);
        try(MemoryWorkspace ws = notifyScopeBorrowed(arrayType)){
            return Nd4j.createUninitialized(shape, order);
        }
    }

    @Override
    public INDArray dup(@NonNull T arrayType, @NonNull INDArray toDup, char order){
        enforceExistsAndActive(arrayType);
        try(MemoryWorkspace ws = notifyScopeBorrowed(arrayType)){
            return toDup.dup(order);
        }
    }

    @Override
    public INDArray dup(@NonNull T arrayType, @NonNull INDArray toDup){
        return dup(arrayType, toDup, toDup.ordering());
    }


    private void validateConfig(@NonNull T arrayType){
        if(scopeOutOfWs.contains(arrayType)){
            return;
        }

        if(!configMap.containsKey(arrayType)){
            throw new ND4JWorkspaceException("No workspace configuration has been provided for arrayType: " + arrayType);
        }
        if(!workspaceNames.containsKey(arrayType)){
            throw new ND4JWorkspaceException("No workspace name has been provided for arrayType: " + arrayType);
        }
    }

    private void enforceExistsAndActive(@NonNull T arrayType){
        validateConfig(arrayType);
        if(scopeOutOfWs.contains(arrayType)){
            return;
        }

        if(!Nd4j.getWorkspaceManager().checkIfWorkspaceExistsAndActive(workspaceNames.get(arrayType))){
            throw new ND4JWorkspaceException("Workspace \"" + workspaceNames.get(arrayType) + "\" for array type " + arrayType
                    + " is not open");
        }
    }
}
