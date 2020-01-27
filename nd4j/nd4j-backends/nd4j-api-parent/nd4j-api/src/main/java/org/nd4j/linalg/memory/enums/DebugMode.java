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
 * This enum describes possible debug modes for Nd4j workspaces
 *
 * @author raver119@protonmail.com
 */
public enum DebugMode {
    /**
     * Default mode, means that workspaces work in production mode
     */
    DISABLED,

    /**
     * All allocations will be considered spilled
     */
    SPILL_EVERYTHING,


    /**
     * All workspaces will be disabled. There will be literally no way to enable workspace anywhere
     */
    BYPASS_EVERYTHING,
}
