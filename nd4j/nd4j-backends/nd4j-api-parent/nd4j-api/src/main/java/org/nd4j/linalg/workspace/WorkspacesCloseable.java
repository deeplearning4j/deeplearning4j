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

import lombok.Data;
import lombok.NonNull;
import org.nd4j.linalg.api.memory.MemoryWorkspace;

/**
 * An {@link AutoCloseable} for multiple workspaces
 * NOTE: Workspaces are closed in REVERSED order to how they are provided to the costructor;
 * this is to mirror nested workspace use
 *
 * @author Alex Black
 */
@Data
public class WorkspacesCloseable implements AutoCloseable {
    private MemoryWorkspace[] workspaces;

    public WorkspacesCloseable(@NonNull MemoryWorkspace... workspaces){
        this.workspaces = workspaces;
    }

    @Override
    public void close() {
        for( int i=workspaces.length-1; i>=0; i-- ){
            workspaces[i].close();
        }
    }
}
