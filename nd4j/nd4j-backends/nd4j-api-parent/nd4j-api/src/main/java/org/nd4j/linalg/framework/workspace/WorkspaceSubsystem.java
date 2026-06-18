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

package org.nd4j.linalg.framework.workspace;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.MemoryWorkspaceManager;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.framework.history.SpillStats;
import org.nd4j.linalg.framework.history.WorkspaceSampler;
import org.nd4j.linalg.framework.stats.WorkspaceStats;

/**
 * Workspace subsystem - provides access to workspace management and tracking.
 * 
 * @author ND4J Team
 */
@Slf4j
public final class WorkspaceSubsystem {
    
    private final WorkspaceEnvironmentAccess environment;
    
    public WorkspaceSubsystem() {
        this.environment = new WorkspaceEnvironmentAccess();
    }
    
    /**
     * Get workspace manager
     */
    public MemoryWorkspaceManager manager() {
        return Nd4j.getWorkspaceManager();
    }
    
    /**
     * Get current workspace
     */
    public MemoryWorkspace currentWorkspace() {
        return Nd4j.getMemoryManager().getCurrentWorkspace();
    }
    
    /**
     * Get workspace statistics
     */
    public WorkspaceStats stats() {
        return WorkspaceStats.builder()
            .activeWorkspaceCount(0)
            .totalWorkspaceBytes(0)
            .totalSpilledBytes(0)
            .totalPinnedBytes(0)
            .totalExternalAllocations(0)
            .build();
    }
    
    /**
     * Get spill statistics
     */
    public SpillStats spillStats() {
        return WorkspaceSampler.getInstance().getSpillStats();
    }
    
    /**
     * Destroy all workspaces for current thread
     */
    public void destroyCurrentThreadWorkspaces() {
        Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
    }
    
    /**
     * Enable workspace sampling
     */
    public void enableSampling() {
        WorkspaceSampler.getInstance().startSampling();
    }
    
    /**
     * Disable workspace sampling
     */
    public void disableSampling() {
        WorkspaceSampler.getInstance().stopSampling();
    }
    
    /**
     * Workspace environment configuration
     */
    public WorkspaceEnvironmentAccess environment() {
        return environment;
    }
    
    // =========================================================================
    // Workspace Environment Access
    // =========================================================================
    
    /**
     * Workspace environment configuration
     */
    public static class WorkspaceEnvironmentAccess {
        
        /**
         * Get default workspace size (bytes)
         */
        public long getDefaultWorkspaceSize() {
            return Long.getLong(org.nd4j.common.config.ND4JSystemProperties.WORKSPACE_DEFAULT_SIZE, 1024 * 1024 * 1024);
        }
        
        /**
         * Set default workspace size (bytes)
         */
        public void setDefaultWorkspaceSize(long sizeBytes) {
            System.setProperty(org.nd4j.common.config.ND4JSystemProperties.WORKSPACE_DEFAULT_SIZE, String.valueOf(sizeBytes));
        }
        
        /**
         * Get workspace initial size (bytes)
         */
        public long getWorkspaceInitialSize() {
            return Long.getLong(org.nd4j.common.config.ND4JSystemProperties.WORKSPACE_INITIAL_SIZE, 1024 * 1024 * 1024);
        }
        
        /**
         * Set workspace initial size (bytes)
         */
        public void setWorkspaceInitialSize(long sizeBytes) {
            System.setProperty(org.nd4j.common.config.ND4JSystemProperties.WORKSPACE_INITIAL_SIZE, String.valueOf(sizeBytes));
        }
        
        /**
         * Check if workspace learning is enabled
         */
        public boolean isWorkspaceLearningEnabled() {
            return Boolean.getBoolean(org.nd4j.common.config.ND4JSystemProperties.WORKSPACE_LEARNING_MODE);
        }
        
        /**
         * Enable/disable workspace learning mode
         */
        public void setWorkspaceLearningEnabled(boolean enabled) {
            System.setProperty(org.nd4j.common.config.ND4JSystemProperties.WORKSPACE_LEARNING_MODE, String.valueOf(enabled));
        }
        
        /**
         * Get workspace debug mode
         */
        public String getWorkspaceDebugMode() {
            return System.getProperty(org.nd4j.common.config.ND4JSystemProperties.WORKSPACE_DEBUG_MODE, "DISABLED");
        }
        
        /**
         * Set workspace debug mode (DISABLED, SPILL_EVERYTHING, CANARY)
         */
        public void setWorkspaceDebugMode(String mode) {
            System.setProperty(org.nd4j.common.config.ND4JSystemProperties.WORKSPACE_DEBUG_MODE, mode);
        }
        
        /**
         * Get summary of workspace environment settings
         */
        public String getSummary() {
            return String.format(
                "Workspace Environment: Default=%dMB, Initial=%dMB, Learning=%s, Debug=%s",
                getDefaultWorkspaceSize() / (1024 * 1024),
                getWorkspaceInitialSize() / (1024 * 1024),
                isWorkspaceLearningEnabled() ? "on" : "off",
                getWorkspaceDebugMode()
            );
        }
    }
}
