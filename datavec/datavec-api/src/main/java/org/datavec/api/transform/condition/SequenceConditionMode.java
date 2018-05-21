/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

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
