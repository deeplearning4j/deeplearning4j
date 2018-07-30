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

package org.nd4j.linalg.memory.abstracts;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.MemoryKind;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This MemoryWorkspace implementation is basically No-Op impl.
 *
 * Do not use it anywhere, unless you 100% sure you need it.
 *
 * @author raver119@gmail.com
 */
public class DummyWorkspace implements MemoryWorkspace {

    protected MemoryWorkspace parentWorkspace;

    /**
     * This method returns WorkspaceConfiguration bean that was used for given Workspace instance
     *
     * @return
     */
    @Override
    public WorkspaceConfiguration getWorkspaceConfiguration() {
        return null;
    }

    /**
     * This method returns Id of this workspace
     *
     * @return
     */
    @Override
    public String getId() {
        return null;
    }

    @Override
    public Long getThreadId() {
        return -1L;
    }

    @Override
    public int getDeviceId() {
        return 0;
    }

    @Override
    public Type getWorkspaceType() {
        return Type.DUMMY;
    }

    /**
     * This method does allocation from a given Workspace
     *
     * @param requiredMemory allocation size, in bytes
     * @param dataType       dataType that is going to be used
     * @param initialize
     * @return
     */
    @Override
    public PagedPointer alloc(long requiredMemory, DataBuffer.Type dataType, boolean initialize) {
        throw new UnsupportedOperationException("DummyWorkspace shouldn't be used for allocation");
    }

    /**
     * This method does allocation from a given Workspace
     *
     * @param requiredMemory allocation size, in bytes
     * @param kind           MemoryKind for allocation
     * @param dataType       dataType that is going to be used
     * @param initialize
     * @return
     */
    @Override
    public PagedPointer alloc(long requiredMemory, MemoryKind kind, DataBuffer.Type dataType, boolean initialize) {
        throw new UnsupportedOperationException("DummyWorkspace shouldn't be used for allocation");
    }

    @Override
    public long getGenerationId() {
        return 0L;
    }

    /**
     * This method notifies given Workspace that new use cycle is starting now
     *
     * @return
     */
    @Override
    public MemoryWorkspace notifyScopeEntered() {
        parentWorkspace = Nd4j.getMemoryManager().getCurrentWorkspace();

        Nd4j.getMemoryManager().setCurrentWorkspace(null);
        return this;
    }

    /**
     * This method TEMPORARY enters this workspace, without reset applied
     *
     * @return
     */
    @Override
    public MemoryWorkspace notifyScopeBorrowed() {
        return null;
    }

    /**
     * This method notifies given Workspace that use cycle just ended
     *
     * @return
     */
    @Override
    public MemoryWorkspace notifyScopeLeft() {
        close();
        return this;
    }

    /**
     * This method returns True if scope was opened, and not closed yet.
     *
     * @return
     */
    @Override
    public boolean isScopeActive() {
        return false;
    }

    /**
     * This method causes Workspace initialization
     * <p>
     * PLEASE NOTE: This call will have no effect on previously initialized Workspace
     */
    @Override
    public void initializeWorkspace() {

    }

    /**
     * This method causes Workspace destruction: all memory allocations are released after this call.
     */
    @Override
    public void destroyWorkspace() {

    }

    @Override
    public void destroyWorkspace(boolean extended) {

    }

    /**
     * This method allows you to temporary disable/enable given Workspace use.
     * If turned off - direct memory allocations will be used.
     *
     * @param isEnabled
     */
    @Override
    public void toggleWorkspaceUse(boolean isEnabled) {

    }

    /**
     * This method returns amount of memory consumed in last successful cycle, in bytes
     *
     * @return
     */
    @Override
    public long getThisCycleAllocations() {
        return 0;
    }

    /**
     * This method enabled debugging mode for this workspace
     *
     * @param reallyEnable
     */
    @Override
    public void enableDebug(boolean reallyEnable) {

    }


    /**
     * This method returns amount of memory consumed in last successful cycle, in bytes
     *
     * @return
     */
    @Override
    public long getLastCycleAllocations() {
        return 0;
    }

    /**
     * This method returns amount of memory consumed by largest successful cycle, in bytes
     *
     * @return
     */
    @Override
    public long getMaxCycleAllocations() {
        return 0;
    }

    /**
     * This methos returns current allocated size of this workspace
     *
     * @return
     */
    @Override
    public long getCurrentSize() {
        return 0;
    }

    @Override
    public void close() {
        Nd4j.getMemoryManager().setCurrentWorkspace(parentWorkspace);
    }

    /**
     * This method returns parent Workspace, if any. Null if there's none.
     *
     * @return
     */
    @Override
    public MemoryWorkspace getParentWorkspace() {
        return null;
    }

    @Override
    public MemoryWorkspace tagOutOfScopeUse() {
        return this;
    }

    @Override
    public void setPreviousWorkspace(MemoryWorkspace memoryWorkspace) {
        parentWorkspace = memoryWorkspace;
    }
}
