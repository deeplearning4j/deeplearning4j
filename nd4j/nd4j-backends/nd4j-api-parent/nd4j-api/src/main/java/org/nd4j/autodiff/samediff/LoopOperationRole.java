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
 * Enumeration of operation roles within loop control flow
 */
public enum LoopOperationRole {
    /**
     * Regular operation that is not part of loop control flow
     */
    REGULAR,
    
    /**
     * Loop condition operation (LoopCond) that determines when loop should terminate
     */
    CONDITION,
    
    /**
     * Exit operation that exits the loop when condition is true
     */
    EXIT,
    
    /**
     * Switch operation that routes values based on predicate
     */
    SWITCH,
    
    /**
     * NextIteration operation that advances to the next loop iteration
     */
    NEXT_ITERATION,
    
    /**
     * Enter operation that enters values into the loop frame
     */
    ENTER,
    
    /**
     * Merge operation that merges values from different control flow paths
     */
    MERGE,
    
    /**
     * Operation that computes loop invariant values
     */
    INVARIANT,
    
    /**
     * Operation that is part of loop initialization
     */
    INITIALIZATION,
    
    /**
     * Operation that is part of loop finalization/cleanup
     */
    FINALIZATION
}
