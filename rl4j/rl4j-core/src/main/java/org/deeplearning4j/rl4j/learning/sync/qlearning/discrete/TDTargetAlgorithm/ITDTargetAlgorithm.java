/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.TDTargetAlgorithm;

import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.nd4j.linalg.dataset.api.DataSet;

import java.util.List;

/**
 * The interface of all TD target calculation algorithms.
 *
 * @param <A> The type of actions
 *
 * @author Alexandre Boulanger
 */
public interface ITDTargetAlgorithm<A> {
    /**
     * Compute the updated estimated Q-Values for every transition
     * @param transitions The transitions from the experience replay
     * @return A DataSet where every element is the observation and the estimated Q-Values for all actions
     */
    DataSet computeTDTargets(List<Transition<A>> transitions);
}
