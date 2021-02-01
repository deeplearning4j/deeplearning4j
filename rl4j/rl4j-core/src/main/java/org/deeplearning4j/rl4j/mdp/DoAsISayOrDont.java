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
package org.deeplearning4j.rl4j.mdp;

import lombok.Getter;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.IntegerActionSchema;
import org.deeplearning4j.rl4j.environment.Schema;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

/**
 * With this environment, the agent is supposed to act exactly as told by the environment -- or the opposite, depending
 * on a randomly switching "do as told / do the opposite" marker in the observation.<br/>
 * Just like {@link TMazeEnvironment}, this environment is designed to be solvable by recurrent networks but unsolvable by non-recurrent ones.
 * But unlike TMaze, which has very sparse rewards, this environment has a reward at every step.<br/>
 * <br/>
 * Format of observations:
 * <ul>
 * <li>Element 1: 1.0 if being told to do action-0 next.
 * <li>Element 2: 1.0 if being told to do action-1 next.
 * <li>Element 3: 1.0 if the agent should do as told, 0.0 if not, and -1.0 if the directive has not changed</li>
 * <li>Element 3: 1.0 if the agent should do the opposite, 0.0 if not, and -1.0 if the directive has not changed</li>
 * </ul>
 */
public class DoAsISayOrDont implements Environment<Integer> {
    private static final int NUM_ACTIONS = 2;

    @Getter
    private final Schema<Integer> schema;
    private final Random rnd;

    private boolean isOpposite;
    private int nextAction;

    public DoAsISayOrDont(Random rnd) {
        this.rnd = rnd != null ? rnd : Nd4j.getRandom();
        this.schema = new Schema<Integer>(new IntegerActionSchema(NUM_ACTIONS, 0, rnd));
    }

    @Override
    public Map<String, Object> reset() {
        nextAction = rnd.nextBoolean() ? 1 : 0;
        isOpposite = rnd.nextBoolean();
        return getChannelsData(true);
    }

    @Override
    public StepResult step(Integer action) {

        double reward;
        if(isOpposite) {
            reward = action != nextAction ? 1.0 : -1.0;
        } else {
            reward = action == nextAction ? 1.0 : -1.0;
        }

        boolean shouldReverse = rnd.nextBoolean();
        if(shouldReverse) {
            isOpposite = !isOpposite;
        }

        return new StepResult(getChannelsData(shouldReverse), reward, false);
    }

    @Override
    public boolean isEpisodeFinished() {
        return false;
    }


    @Override
    public void close() {

    }

    private Map<String, Object> getChannelsData(boolean showIndicators) {
        double normalModeIndicator = showIndicators
                ? (isOpposite ? 0.0 : 1.0)
                : -1.0;
        double oppositeModeIndicator = showIndicators
                ? (isOpposite ? 1.0 : 0.0)
                : -1.0;

        return new HashMap<String, Object>() {{
            put("data", new double[]{ (double)nextAction, (1.0 - nextAction), normalModeIndicator, oppositeModeIndicator});
        }};
    }
}
