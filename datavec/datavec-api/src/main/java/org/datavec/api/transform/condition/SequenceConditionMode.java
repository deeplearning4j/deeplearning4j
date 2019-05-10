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

package org.datavec.api.transform.condition;

/**
 * For certain single-column conditions: how should we apply these to sequences?<br>
 * <b>And</b>: Condition applies to sequence only if it applies to ALL time steps<br>
 * <b>Or</b>: Condition applies to sequence if it applies to ANY time steps<br>
 * <b>NoSequencMode</b>: Condition cannot be applied to sequences at all (error condition)
 *
 * @author Alex Black
 */
public enum SequenceConditionMode {
    And, Or, NoSequenceMode
}
