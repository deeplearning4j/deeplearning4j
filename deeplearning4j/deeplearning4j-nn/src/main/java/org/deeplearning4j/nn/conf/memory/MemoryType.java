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

package org.deeplearning4j.nn.conf.memory;

/**
 * Type of memory
 *
 * @author Alex Black
 */
public enum MemoryType {
    PARAMETERS, PARAMATER_GRADIENTS, ACTIVATIONS, ACTIVATION_GRADIENTS, UPDATER_STATE, WORKING_MEMORY_FIXED, WORKING_MEMORY_VARIABLE, CACHED_MEMORY_FIXED, CACHED_MEMORY_VARIABLE;

    /**
     * @return True, if the memory type is used during inference. False if the memory type is used only during training.
     */
    public boolean isInference() {
        switch (this) {
            case PARAMETERS:
            case ACTIVATIONS:
            case WORKING_MEMORY_FIXED:
            case WORKING_MEMORY_VARIABLE:
                return true;
            case PARAMATER_GRADIENTS:
            case ACTIVATION_GRADIENTS:
            case UPDATER_STATE:
            case CACHED_MEMORY_FIXED:
            case CACHED_MEMORY_VARIABLE:
                return false;
        }
        throw new RuntimeException("Unknown memory type: " + this);
    }
}
