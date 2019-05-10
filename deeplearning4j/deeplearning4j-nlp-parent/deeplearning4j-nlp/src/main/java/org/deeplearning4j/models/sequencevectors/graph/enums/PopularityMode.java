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
 * This enum is used in PopularityWalker, and it defines which nodes will be considered for next hop.
 * MAXIMUM: top-popularity nodes will be considered.
 * AVERAGE: nodes in the middle of possible selections will be considered.
 * MINIMUM: low-popularity nodes will be considered.
 *
 * @author raver119@gmail.com
 */
public enum PopularityMode {
    MAXIMUM, AVERAGE, MINIMUM,
}
