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

package org.deeplearning4j.nn.workspace;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.base.Preconditions;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.workspace.BaseWorkspaceMgr;
import org.nd4j.linalg.workspace.WorkspaceMgr;

import java.util.*;

public class LayerWorkspaceMgr extends BaseWorkspaceMgr<ArrayType> {

    private static LayerWorkspaceMgr NO_WS_IMMUTABLE;
    static {
        Set<ArrayType> all = new HashSet<>();
        Collections.addAll(all, ArrayType.values());
        NO_WS_IMMUTABLE = new LayerWorkspaceMgr(
                all, Collections.emptyMap(), Collections.emptyMap());
    }

    protected Set<String> noLeverageOverride;

    @Setter @Getter
    protected Map<String,Pointer> helperWorkspacePointers;

    private LayerWorkspaceMgr() {

    }

    public LayerWorkspaceMgr(Set<ArrayType> scopeOutOfWs, Map<ArrayType, WorkspaceConfiguration> configMap,
                             Map<ArrayType, String> workspaceNames) {
        super(scopeOutOfWs, configMap, workspaceNames);
        if(configMap != null) {
            Preconditions.checkArgument(configMap.keySet().equals(workspaceNames.keySet()),
                    "Keys for config may and workspace names must match");
        }
    }

    public void setNoLeverageOverride(String wsName) {
        if(noLeverageOverride == null) {
            noLeverageOverride = new HashSet<>();
        }
        noLeverageOverride.add(wsName);
    }

    @Override
    public INDArray leverageTo(ArrayType arrayType, INDArray array) {
        if(noLeverageOverride != null && array.isAttached() && noLeverageOverride.contains(array.data().getParentWorkspace().getId())) {
            return array;
        }
        return super.leverageTo(arrayType, array);
    }

    @Override
    public INDArray validateArrayLocation(@NonNull ArrayType arrayType, @NonNull INDArray array, boolean migrateIfInvalid, boolean exceptionIfDetached) {
        if(noLeverageOverride != null && array.isAttached() && noLeverageOverride.contains(array.data().getParentWorkspace().getId())) {
            return array;   //OK - leverage override
        }
        return super.validateArrayLocation(arrayType, array, migrateIfInvalid, exceptionIfDetached);
    }



    public static Builder builder(){
        return new Builder();
    }

    /**
     * @param helperWorkspacePointers Helper pointers - see {@link #getHelperWorkspace(String)} for details
     * @return Workspace manager
     */
    public static LayerWorkspaceMgr noWorkspaces(Map<String,Pointer> helperWorkspacePointers) {
        LayerWorkspaceMgr wsm = noWorkspaces();
        wsm.setHelperWorkspacePointers(helperWorkspacePointers);
        return wsm;
    }

    public static LayerWorkspaceMgr noWorkspaces(){
        return builder().defaultNoWorkspace().build();
    }

    public static LayerWorkspaceMgr noWorkspacesImmutable(){
        return NO_WS_IMMUTABLE;
    }

    public static class Builder {

        private LayerWorkspaceMgr mgr;

        public Builder(){
            mgr = new LayerWorkspaceMgr();
        }

        /**
         * Set the default to be scoped out for all array types.
         * NOTE: Will not override the configuration for any array types that have already been configured
         * @return Builder
         */
        public Builder defaultNoWorkspace() {
            for(ArrayType t : ArrayType.values()) {
                if(!mgr.configMap.containsKey(t)) {
                    mgr.setScopedOutFor(t);
                }
            }
            return this;
        }

        /**
         * Specify that no workspace should be used for array of the specified type - i.e., these arrays should all
         * be scoped out.
         *
         * @param type Array type to set scoped out for
         * @return Builder
         */
        public Builder noWorkspaceFor(ArrayType type) {
            mgr.setScopedOutFor(type);
            return this;
        }

        /**
         * Set the default workspace for all array types to the specified workspace name/configuration
         * NOTE: This will NOT override any settings previously set.
         *
         * @param workspaceName Name of the workspace to use for all (not set) array types
         * @param configuration Configuration to use for all (not set) array types
         * @return Builder
         */
        public Builder defaultWorkspace(String workspaceName, WorkspaceConfiguration configuration) {
            for(ArrayType t : ArrayType.values()) {
                if(!mgr.configMap.containsKey(t) && !mgr.isScopedOut(t)) {
                    with(t, workspaceName, configuration);
                }
            }
            return this;
        }

        /**
         * Configure the workspace (name, configuration) for the specified array type
         *
         * @param type          Array type
         * @param workspaceName Workspace name for the specified array type
         * @param configuration Configuration for the specified array type
         * @return Builder
         */
        public Builder with(ArrayType type, String workspaceName, WorkspaceConfiguration configuration) {
            mgr.setConfiguration(type, configuration);
            mgr.setWorkspaceName(type, workspaceName);
            return this;
        }

        public LayerWorkspaceMgr build() {
            return mgr;
        }

    }

    public  List<org.deeplearning4j.nn.workspace.ArrayType> allOpen() {
        List<org.deeplearning4j.nn.workspace.ArrayType> list = new ArrayList<>();
        for(org.deeplearning4j.nn.workspace.ArrayType t : org.deeplearning4j.nn.workspace.ArrayType.values()) {
            String name = this.getWorkspaceName(t);
            if(name != null && Nd4j.getWorkspaceManager().checkIfWorkspaceExistsAndActive(name)) {
                list.add(t);
            }
        }
        return list;
    }


}
