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

package org.deeplearning4j.nn.conf;

/**
 * Workspace mode to use. See https://deeplearning4j.org/workspaces<br>
 * <br>
 * NONE: No workspaces will be used for the network. Highest memory use, least performance.<br>
 * ENABLED: Use workspaces.<br>
 * SINGLE: Deprecated. Now equivalent to ENABLED, which should be used instead.<br>
 * SEPARATE: Deprecated. Now equivalent to ENABLED, which sohuld be used instead.<br>
 *
 * @author raver119@gmail.com
 */
public enum WorkspaceMode {
    NONE, // workspace won't be used
    ENABLED,
    /**
     * @deprecated Use {@link #ENABLED} instead
     */
    @Deprecated
    SINGLE, // one external workspace
    /**
     * @deprecated Use {@link #ENABLED} instead
     */
    @Deprecated
    SEPARATE, // one external workspace, one FF workspace, one BP workspace <-- default one
}
