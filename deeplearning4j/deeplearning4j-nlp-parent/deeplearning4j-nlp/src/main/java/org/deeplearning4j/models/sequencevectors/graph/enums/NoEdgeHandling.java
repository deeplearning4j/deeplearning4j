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

package org.deeplearning4j.models.sequencevectors.graph.enums;

/**
 * This enum describes different behaviors for cases when GraphWalker don't have next hop within current walk.
 *
 * @author raver119@gmail.com
 */
public enum NoEdgeHandling {
    SELF_LOOP_ON_DISCONNECTED, EXCEPTION_ON_DISCONNECTED, PADDING_ON_DISCONNECTED, CUTOFF_ON_DISCONNECTED, RESTART_ON_DISCONNECTED,
}
