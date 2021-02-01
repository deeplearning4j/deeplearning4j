/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.IPolicy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

public class MockPolicy implements IPolicy<Integer> {

    public int playCallCount = 0;
    public List<INDArray> actionInputs = new ArrayList<INDArray>();

    @Override
    public <MockObservation extends Encodable, AS extends ActionSpace<Integer>> double play(MDP<MockObservation, Integer, AS> mdp, IHistoryProcessor hp) {
        ++playCallCount;
        return 0;
    }

    @Override
    public Integer nextAction(INDArray input) {
        actionInputs.add(input);
        return input.getInt(0, 0, 0);
    }

    @Override
    public Integer nextAction(Observation observation) {
        return nextAction(observation.getData());
    }

    @Override
    public void reset() {

    }
}
