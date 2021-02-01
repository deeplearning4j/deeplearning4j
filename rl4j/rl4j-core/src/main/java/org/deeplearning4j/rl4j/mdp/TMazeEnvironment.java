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

package org.deeplearning4j.rl4j.mdp;

import lombok.Getter;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.IntegerActionSchema;
import org.deeplearning4j.rl4j.environment.Schema;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.nd4j.linalg.api.rng.Random;

import java.util.HashMap;
import java.util.Map;

/**
 * This environment is a non-Markovian grid-based T-maze like this: <pre>
 *                                       +---+
 *                                       | X |
 * +---+---+---+---+     +---+---+---+---+---+
 * | S |   |   |   | ... |   |   |   |   | B |
 * +---+---+---+---+     +---+---+---+---+---+
 *                                       | Y |
 *                                       +---+
 * </pre>
 * The agent start at cell 'S' and must navigate to the <i>goal</i>. The location of this goal is selected randomly between
 * cell X and cell Y at the start of each episode.<br />
 * This environment is designed to be straightforward except for one aspect: Only the initial observation contains
 * information on the location of the goal. This means that the agent must remember the location of the goal while
 * navigating the t-maze.<br />
 * This makes the environment unsolvable with networks having only dense layers, but solvable with RNNs.
 * <p />
 * A reward of 4.0 is returned when reaching the goal, and -4.0 when reaching the trap. Reaching the goal or the trap
 * ends the episode.<p />
 * Also, to make the agent learn to navigate to the branch faster, a reward of -0.1 is returned each time the agent bumps
 * into a wall or navigates left (away from the branch). And a one-time (per episode) reward of 1.0 is returned when the
 * agent reaches the branch.<br />
 * <br />
 * Format of observations:
 * <ul>
 * <li>Element 1: 1.0 if it's the first observation; otherwise 0.0</li>
 * <li>Element 2: 1.0 if it's not the first observation and the agent is not at the branch; otherwise 0.0</li>
 * <li>Element 3: 1.0 if the agent is at the branch; otherwise 0.0</li>
 * <li>Element 4: 1.0 if the goal is at cell 'X', 0.0 if not, and -1.0 if the location of the goal is not observable</li>
 * <li>Element 5: 1.0 if the goal is at cell 'Y', 0.0 if not, and -1.0 if the location of the goal is not observable</li>
 * </ul>
 */
public class TMazeEnvironment implements Environment<Integer> {
    private static final double BAD_MOVE_REWARD = -0.1;
    private static final double GOAL_REWARD = 4.0;
    private static final double TRAP_REWARD = -4.0;
    private static final double BRANCH_REWARD = 1.0;

    private static final int NUM_ACTIONS = 4;
    private static final int ACTION_LEFT = 0;
    private static final int ACTION_RIGHT = 1;
    private static final int ACTION_UP = 2;
    private static final int ACTION_DOWN = 3;

    private final int lengthOfMaze;
    private final Random rnd;

    @Getter
    private final Schema<Integer> schema;

    private int currentLocation;
    private boolean hasNavigatedToBranch;

    private boolean hasNavigatedToSolution;
    public boolean hasNavigatedToSolution() {
        return hasNavigatedToSolution;
    }

    private boolean isSolutionUp;

    @Getter
    boolean episodeFinished;

    public TMazeEnvironment(int lengthOfMaze, Random rnd) {
        this.lengthOfMaze = lengthOfMaze;
        this.rnd = rnd;

        this.schema = new Schema<Integer>(new IntegerActionSchema(NUM_ACTIONS, ACTION_RIGHT, rnd));
    }

    @Override
    public Map<String, Object> reset() {
        episodeFinished = false;
        currentLocation = 0;
        hasNavigatedToBranch = false;

        isSolutionUp = rnd.nextBoolean();

        return new HashMap<String, Object>() {{
            put("data", new double[] { 1.0, 0.0, 0.0, isSolutionUp ? 1.0 : 0.0, isSolutionUp ? 0.0 : 1.0 });
        }};
    }

    @Override
    public StepResult step(Integer action) {
        boolean isAtJunction = currentLocation == lengthOfMaze - 1;
        double reward = 0.0;

        if (!episodeFinished) {
            switch (action) {
                case ACTION_LEFT:
                    reward = BAD_MOVE_REWARD;
                    if(currentLocation > 0) {
                        --currentLocation;
                    }
                    break;

                case ACTION_RIGHT:
                    if(isAtJunction) {
                        reward = BAD_MOVE_REWARD;
                    } else {
                        ++currentLocation;
                    }
                    break;

                case ACTION_UP:
                    if(!isAtJunction) {
                        reward = BAD_MOVE_REWARD;
                    } else {
                        reward = isSolutionUp ? GOAL_REWARD : TRAP_REWARD;
                        hasNavigatedToSolution = isSolutionUp;
                        episodeFinished = true;
                    }
                    break;

                case ACTION_DOWN:
                    if(!isAtJunction) {
                        reward = BAD_MOVE_REWARD;
                    } else {
                        reward = !isSolutionUp ? GOAL_REWARD : TRAP_REWARD;
                        hasNavigatedToSolution = !isSolutionUp;
                        episodeFinished = true;
                    }
                    break;
            }
        }

        boolean isAtJunctionAfterMove = currentLocation == lengthOfMaze - 1;
        if(!hasNavigatedToBranch && isAtJunctionAfterMove) {
            reward += BRANCH_REWARD;
            hasNavigatedToBranch = true;
        }
        double[] channelData = isAtJunctionAfterMove
                ? new double[] { 0.0, 0.0, 1.0, -1.0, -1.0 }
                : new double[] { 0.0, 1.0, 0.0, -1.0, -1.0 };

        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("data", channelData);
        }};
        return new StepResult(channelsData, reward, episodeFinished);
    }


    @Override
    public void close() {
        // Do nothing
    }
}