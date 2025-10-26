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
 * Entity type enumeration for SameDiff graph elements.
 * Used to classify different types of entities in the computation graph.
 */
public enum EntityType {
    /**
     * Operation in the computation graph
     */
    OPERATION,
    
    /**
     * Variable in the computation graph (including outputs of operations)
     */
    VARIABLE,
    
    /**
     * Constant value in the computation graph
     */
    CONSTANT,
    
    /**
     * Placeholder that requires runtime input
     */
    PLACEHOLDER,
    
    /**
     * Unknown or unclassified entity
     */
    UNKNOWN
}
