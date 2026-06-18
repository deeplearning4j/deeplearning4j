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

import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Workspace subsystem environment configuration.
 * 
 * @author ND4J Team
 */
public final class WorkspaceEnvironment {
    
    private WorkspaceEnvironment() {}
    
    /**
     * Get default workspace size (bytes)
     */
    public long getDefaultWorkspaceSize() {
        return Long.getLong(ND4JSystemProperties.WORKSPACE_DEFAULT_SIZE, 1024 * 1024 * 1024);
    }
    
    /**
     * Set default workspace size (bytes)
     */
    public void setDefaultWorkspaceSize(long sizeBytes) {
        System.setProperty(ND4JSystemProperties.WORKSPACE_DEFAULT_SIZE, String.valueOf(sizeBytes));
    }
    
    /**
     * Get workspace initial size (bytes)
     */
    public long getWorkspaceInitialSize() {
        return Long.getLong(ND4JSystemProperties.WORKSPACE_INITIAL_SIZE, 1024 * 1024 * 1024);
    }
    
    /**
     * Set workspace initial size (bytes)
     */
    public void setWorkspaceInitialSize(long sizeBytes) {
        System.setProperty(ND4JSystemProperties.WORKSPACE_INITIAL_SIZE, String.valueOf(sizeBytes));
    }
    
    /**
     * Get workspace maximum size (bytes)
     */
    public long getWorkspaceMaxSize() {
        return Long.getLong(ND4JSystemProperties.WORKSPACE_MAX_SIZE, 4L * 1024 * 1024 * 1024);
    }
    
    /**
     * Set workspace maximum size (bytes)
     */
    public void setWorkspaceMaxSize(long sizeBytes) {
        System.setProperty(ND4JSystemProperties.WORKSPACE_MAX_SIZE, String.valueOf(sizeBytes));
    }
    
    /**
     * Check if workspace learning is enabled
     */
    public boolean isWorkspaceLearningEnabled() {
        return Boolean.getBoolean(ND4JSystemProperties.WORKSPACE_LEARNING_MODE);
    }
    
    /**
     * Enable/disable workspace learning mode
     */
    public void setWorkspaceLearningEnabled(boolean enabled) {
        System.setProperty(ND4JSystemProperties.WORKSPACE_LEARNING_MODE, String.valueOf(enabled));
    }
    
    /**
     * Get workspace debug mode
     */
    public String getWorkspaceDebugMode() {
        return System.getProperty(ND4JSystemProperties.WORKSPACE_DEBUG_MODE, "DISABLED");
    }
    
    /**
     * Set workspace debug mode (DISABLED, SPILL_EVERYTHING, CANARY)
     */
    public void setWorkspaceDebugMode(String mode) {
        System.setProperty(ND4JSystemProperties.WORKSPACE_DEBUG_MODE, mode);
    }
    
    /**
     * Check if workspace pre-allocation is enabled
     */
    public boolean isWorkspacePreallocate() {
        return Boolean.getBoolean(ND4JSystemProperties.WORKSPACE_PREALLOCATE);
    }
    
    /**
     * Enable/disable workspace pre-allocation
     */
    public void setWorkspacePreallocate(boolean enabled) {
        System.setProperty(ND4JSystemProperties.WORKSPACE_PREALLOCATE, String.valueOf(enabled));
    }
    
    /**
     * Get workspace overallocation limit
     */
    public double getWorkspaceOverallocationLimit() {
        String val = System.getProperty(ND4JSystemProperties.WORKSPACE_OVERALLOCATION_LIMIT);
        if (val == null) return 1.5;
        try {
            return Double.parseDouble(val);
        } catch (NumberFormatException e) {
            return 1.5;
        }
    }
    
    /**
     * Set workspace overallocation limit (e.g., 1.5 = 150% of initial size)
     */
    public void setWorkspaceOverallocationLimit(double limit) {
        System.setProperty(ND4JSystemProperties.WORKSPACE_OVERALLOCATION_LIMIT, String.valueOf(limit));
    }
    
    /**
     * Get summary of workspace environment settings
     */
    public String getSummary() {
        return String.format(
            "Workspace Environment: Default=%dMB, Initial=%dMB, Max=%dMB, " +
            "Learning=%s, Debug=%s, Preallocate=%s, Overallocation=%.1fx",
            getDefaultWorkspaceSize() / (1024 * 1024),
            getWorkspaceInitialSize() / (1024 * 1024),
            getWorkspaceMaxSize() / (1024 * 1024),
            isWorkspaceLearningEnabled() ? "on" : "off",
            getWorkspaceDebugMode(),
            isWorkspacePreallocate() ? "on" : "off",
            getWorkspaceOverallocationLimit()
        );
    }
}
