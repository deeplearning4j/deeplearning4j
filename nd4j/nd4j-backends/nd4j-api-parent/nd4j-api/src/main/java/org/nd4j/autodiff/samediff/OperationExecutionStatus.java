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
 * Enumeration of possible operation execution statuses
 */
public enum OperationExecutionStatus {
    /**
     * Operation has not been executed yet
     */
    NOT_EXECUTED,
    
    /**
     * Operation is currently being executed
     */
    EXECUTING,
    
    /**
     * Operation executed successfully
     */
    SUCCESS,
    
    /**
     * Operation was analyzed (values captured but not necessarily executed)
     */
    ANALYZED,
    
    /**
     * Operation execution failed with an error
     */
    ERROR,
    
    /**
     * Operation was skipped during execution
     */
    SKIPPED,
    
    /**
     * Operation execution was cancelled
     */
    CANCELLED,
    
    /**
     * Operation execution timed out
     */
    TIMEOUT,
    
    /**
     * Operation status is unknown
     */
    UNKNOWN
}
