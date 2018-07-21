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

package org.nd4j.linalg.api.memory.enums;

/**
 * @author raver119@gmail.com
 */
public enum LearningPolicy {
    /**
     * This policy means - we learn during 1 cycle,
     * and allocate workspace memory right after it's done.
     */
    FIRST_LOOP,

    /**
     * This policy means - we learn during multiple cycles,
     * and allocate after WorkspaceConfiguration.cyclesBeforeInitialization
     * or after manual call to MemoryWorkspace.initializeWorkspace
     */
    OVER_TIME,

    /**
     * This policy means - no learning is assumed, WorkspaceConfiguration.initialSize value will be primary determinant for workspace size
     */
    NONE,
}
