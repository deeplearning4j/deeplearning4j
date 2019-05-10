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

package org.deeplearning4j.arbiter.task;

import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.nn.api.Model;

import java.io.Serializable;

/**
 * TaskListener: can be used to preprocess and post process a model (MultiLayerNetwork or ComputationGraph) before/after
 * training, in a {@link MultiLayerNetworkTaskCreator} or {@link ComputationGraphTaskCreator}
 *
 * @author Alex Black
 */
public interface TaskListener extends Serializable {

    /**
     * Preprocess the model, before any training has taken place.
     * <br>
     * Can be used to (for example) set listeners on a model before training starts
     * @param model     Model to preprocess
     * @param candidate Candidate information, for the current model
     * @return The updated model (usually the same one as the input, perhaps with modifications)
     */
    <T extends Model> T preProcess(T model, Candidate candidate);

    /**
     * Post process the model, after any training has taken place
     * @param model     Model to postprocess
     * @param candidate Candidate information, for the current model
     */
    void postProcess(Model model, Candidate candidate);

}
