/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff;

/**
 * An SDVarible may have different uses in a SameDiff graph - VariableType represents these different roles/uses.<br>
 * <br>
 * VARIABLE: an array of trainable parameters in the SameDiff instance. Must be floating point (to be trainable by backprop)<br>
 * CONSTANT: a fixed value that should not be modified during training/inference. May be replaced by the user.<br>
 * ARRAY: an intermediate array (ie., output of an op) that is not trainable and is usually not persisted. For example, activations.<br>
 * PLACEHOLDER: represents an array to be provided later. Your input features and labels are placeholders.<br>
 * <br>
 * <br>
 * <pre>
 * Type         Trainable   Gradients       Persisted       Workspaces      DataTypes
 * VARIABLE     Yes         Yes             Yes             No              Floating point only
 * CONSTANT     No          No              Yes             No              All
 * ARRAY        No          Yes             No              Yes             All
 * PLACEHOLDER  No          No              No              No              All
 * </pre>
 *
 */
public enum VariableType {
    VARIABLE,
    CONSTANT,
    ARRAY,
    PLACEHOLDER
}
