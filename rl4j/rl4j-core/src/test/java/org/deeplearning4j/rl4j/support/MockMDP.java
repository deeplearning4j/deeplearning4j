/*
 *  ******************************************************************************
 *  *
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

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.transform.EncodableToINDArrayTransform;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.observation.transform.filter.UniformSkippingFilter;
import org.deeplearning4j.rl4j.observation.transform.operation.HistoryMergeTransform;
import org.deeplearning4j.rl4j.observation.transform.operation.SimpleNormalizationTransform;
import org.deeplearning4j.rl4j.observation.transform.operation.historymerge.CircularFifoStore;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.rng.Random;

import java.util.ArrayList;
import java.util.List;

public class MockMDP implements MDP<MockObservation, Integer, DiscreteSpace> {

    private final DiscreteSpace actionSpace;
    private final int stepsUntilDone;
    private int currentObsValue = 0;
    private final ObservationSpace observationSpace;

    public final List<Integer> actions = new ArrayList<>();
    private int step = 0;
    public int resetCount = 0;

    public MockMDP(ObservationSpace observationSpace, int stepsUntilDone, DiscreteSpace actionSpace) {
        this.stepsUntilDone = stepsUntilDone;
        this.actionSpace = actionSpace;
        this.observationSpace = observationSpace;
    }

    public MockMDP(ObservationSpace observationSpace, int stepsUntilDone, Random rnd) {
        this(observationSpace, stepsUntilDone, new DiscreteSpace(5, rnd));
    }

    public MockMDP(ObservationSpace observationSpace) {
        this(observationSpace, Integer.MAX_VALUE, new DiscreteSpace(5));
    }

    public MockMDP(ObservationSpace observationSpace, Random rnd) {
        this(observationSpace, Integer.MAX_VALUE, new DiscreteSpace(5, rnd));
    }

    @Override
    public ObservationSpace getObservationSpace() {
        return observationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return actionSpace;
    }

    @Override
    public MockObservation reset() {
        ++resetCount;
        currentObsValue = 0;
        step = 0;
        return new MockObservation(currentObsValue++);
    }

    @Override
    public void close() {

    }

    @Override
    public StepReply<MockObservation> step(Integer action) {
        actions.add(action);
        ++step;
        return new StepReply<>(new MockObservation(currentObsValue), (double) currentObsValue++, isDone(), null);
    }

    @Override
    public boolean isDone() {
        return step >= stepsUntilDone;
    }

    @Override
    public MDP newInstance() {
        return null;
    }

    public static TransformProcess buildTransformProcess(int skipFrame, int historyLength) {
        return TransformProcess.builder()
                .filter(new UniformSkippingFilter(skipFrame))
                .transform("data", new EncodableToINDArrayTransform())
                .transform("data", new SimpleNormalizationTransform(0.0, 255.0))
                .transform("data", HistoryMergeTransform.builder()
                        .elementStore(new CircularFifoStore(historyLength))
                        .build(4))
                .build("data");
    }

}
