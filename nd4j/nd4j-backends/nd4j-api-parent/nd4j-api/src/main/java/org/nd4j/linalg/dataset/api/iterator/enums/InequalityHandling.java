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

package org.nd4j.linalg.dataset.api.iterator.enums;

/**
 * This enum describes different handling options for situations once one of producer runs out of data
 *
 * @author raver119@gmail.com
 */
public enum InequalityHandling {
    /**
     * Parallel iterator will stop everything once one of producers runs out of data
     */
    STOP_EVERYONE,

    /**
     * Parallel iterator will keep returning true on hasNext(), but next() will return null instead of DataSet
     */
    PASS_NULL,

    /**
     * Parallel iterator will silently reset underlying producer
     */
    RESET,

    /**
     * Parallel iterator will ignore this producer, and will use other producers.
     *
     * PLEASE NOTE: This option will invoke cross-device relocation in multi-device systems.
     */
    RELOCATE,
}
