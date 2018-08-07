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
public enum SpillPolicy {
    /**
     * This policy means - use external allocation for spills.
     *
     * PLEASE NOTE: external allocations will be released after end of loop is reached.
     */
    EXTERNAL,

    /**
     * This policy means - use external allocation for spills + reallocate at the end of loop.
     */
    REALLOCATE,

    /**
     * This policy means - no spills will be ever possible, exception will be thrown.
     *
     * PLEASE NOTE: basically useful for debugging.
     */
    FAIL,
}
