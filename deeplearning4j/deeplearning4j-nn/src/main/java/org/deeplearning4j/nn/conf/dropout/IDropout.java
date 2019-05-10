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

package org.deeplearning4j.nn.conf.dropout;

import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * IDropout instances operate on an activations array, modifying or dropping values at training time only.
 * IDropout instances are not applied at test time.
 *
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface IDropout extends Serializable, Cloneable {

    /**
     * @param inputActivations Input activations array
     * @param resultArray      The result array (same as inputArray for in-place ops) for the post-dropout activations
     * @param iteration        Current iteration number
     * @param epoch            Current epoch number
     * @param workspaceMgr     Workspace manager, if any storage is required (use ArrayType.INPUT)
     * @return The output (resultArray) after applying dropout
     */
    INDArray applyDropout(INDArray inputActivations, INDArray resultArray, int iteration, int epoch, LayerWorkspaceMgr workspaceMgr);

    /**
     * Perform backprop. This should also clear the internal state (dropout mask) if any is present
     *
     * @param gradAtOutput Gradients at the output of the dropout op - i.e., dL/dOut
     * @param gradAtInput  Gradients at the input of the dropout op - i.e., dL/dIn. Use the same array as gradAtOutput
     *                     to apply the backprop gradient in-place
     * @param iteration    Current iteration
     * @param epoch        Current epoch
     * @return Same array as gradAtInput - i.e., gradient after backpropagating through dropout op - i.e., dL/dIn
     */
    INDArray backprop(INDArray gradAtOutput, INDArray gradAtInput, int iteration, int epoch);

    /**
     * Clear the internal state (for example, dropout mask) if any is present
     */
    void clear();

    IDropout clone();
}
