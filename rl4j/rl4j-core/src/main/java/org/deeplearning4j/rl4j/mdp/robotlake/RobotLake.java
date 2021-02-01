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
package org.deeplearning4j.rl4j.mdp.robotlake;

import lombok.Getter;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.IntegerActionSchema;
import org.deeplearning4j.rl4j.environment.Schema;
import org.deeplearning4j.rl4j.environment.StepResult;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

/**
 * RobotLake is a spin off of FrozenLake. Most of it is the same except that it is a robot that tries to reach the
 * goal on the lake. And instead of observing the whole grid, the robot has a 'radar' that only sees what is
 * directly up, right, down and left relative to his position, and a 'tracker' that informs the robot of the horizontal and
 * vertical distance to the goal.
 * <br/>
 * This environment is designed to easily merge the observations into a single channel for comparison
 * <br/>
 * Format of observations:<br/>
 * Channel <i>tracker</i>:
 * <ul>
 * <li>Element 1: Signed vertical distance from the robot to the goal. negative means the goal is up; positive, the goal is down
 * <li>Element 2: Signed horizontal distance from the robot to the goal. negative means the goal is left; positive, the goal is right
 * </ul>
 * <br/>
 * Channel <i>radar</i>:
 * <ul>
 * <li>Element 1: 1.0 means the cell in the up direction is safe; 0.0 otherwise
 * <li>Element 2: 1.0 means the cell in the right direction is safe; 0.0 otherwise
 * <li>Element 3: 1.0 means the cell in the down direction is safe; 0.0 otherwise
 * <li>Element 4: 1.0 means the cell in the left direction is safe; 0.0 otherwise
 * </ul>
 */
public class RobotLake implements Environment<Integer>  {
    private static final double GOAL_REWARD = 10.0;
    private static final double STEPPED_ON_HOLE_REWARD = -2.0;
    private static final double MOVE_AWAY_FROM_GOAL_REWARD = -0.1;

    public static final int NUM_ACTIONS = 4;
    public static final int ACTION_LEFT = 0;
    public static final int ACTION_RIGHT = 1;
    public static final int ACTION_UP = 2;
    public static final int ACTION_DOWN = 3;

    public static final char PLAYER = 'P';
    public static final char GOAL = 'G';
    public static final char HOLE = '@';
    public static final char ICE = ' ';

    @Getter
    private Schema<Integer> schema;

    @Getter
    private boolean episodeFinished = false;

    @Getter
    private boolean goalReached = false;

    @Getter
    private DiscreteSpace actionSpace = new DiscreteSpace(NUM_ACTIONS);

    @Getter
    private ObservationSpace<RobotLakeState> observationSpace = new ArrayObservationSpace(new int[] {  });

    private RobotLakeState state;
    private final int size;

    public RobotLake(int size) {
        this(size, false, Nd4j.getRandom());
    }

    public RobotLake(int size, boolean areStartingPositionsRandom, Random rnd) {
        state = new RobotLakeState(size, areStartingPositionsRandom, rnd);
        this.size = size;
        this.schema = new Schema<Integer>(new IntegerActionSchema(NUM_ACTIONS, ACTION_LEFT, rnd));
    }

    @Override
    public Map<String, Object> reset() {
        state.reset();
        episodeFinished = false;
        goalReached = false;

        return getChannelsData();
    }

    public StepResult step(Integer action) {
        double reward = 0.0;

        switch (action) {
            case ACTION_LEFT:
                state.moveRobotLeft();
                break;

            case ACTION_RIGHT:
                state.moveRobotRight();
                break;

            case ACTION_UP:
                state.moveRobotUp();
                break;

            case ACTION_DOWN:
                state.moveRobotDown();
                break;
        }

        if(RobotLakeHelper.isGoalAtLocation(state.getLake(), state.getRobotY(), state.getRobotX())) {
            episodeFinished = true;
            goalReached = true;
            reward = GOAL_REWARD;
        } else if(!RobotLakeHelper.isLocationSafe(state.getLake(), state.getRobotY(), state.getRobotX())) {
            episodeFinished = true;
            reward = STEPPED_ON_HOLE_REWARD;
        } else {
            // Give a small negative reward for moving away from the goal (to speedup learning)
            switch (action) {
                case ACTION_LEFT:
                    reward = state.getGoalX() > 0 ? MOVE_AWAY_FROM_GOAL_REWARD : 0.0;
                    break;

                case ACTION_RIGHT:
                    reward = state.getGoalX() == 0 ? MOVE_AWAY_FROM_GOAL_REWARD : 0.0;
                    break;

                case ACTION_UP:
                    reward = state.getGoalY() > 0 ? MOVE_AWAY_FROM_GOAL_REWARD : 0.0;
                    break;

                case ACTION_DOWN:
                    reward = state.getGoalY() == 0 ? MOVE_AWAY_FROM_GOAL_REWARD : 0.0;
                    break;
            }
        }

        return new StepResult(getChannelsData(), reward, episodeFinished);
    }


    @Override
    public void close() {
        // Do nothing
    }

    private double[] getTrackerChannelData() {
        return new double[] {
                state.getGoalY() - state.getRobotY(),
                state.getGoalX() - state.getRobotX()
        };
    }

    private double[] getRadarChannelData() {
        return new double[] {
                // UP Direction
                state.getRobotY() == 0  || RobotLakeHelper.isLocationSafe(state.getLake(), state.getRobotY() - 1, state.getRobotX()) ? 1.0 : 0.0,

                // RIGHT Direction
                state.getRobotX() == (size - 1) || RobotLakeHelper.isLocationSafe(state.getLake(), state.getRobotY(), state.getRobotX() + 1) ? 1.0 : 0.0,

                // DOWN Direction
                state.getRobotY() == (size - 1) || RobotLakeHelper.isLocationSafe(state.getLake(), state.getRobotY() + 1, state.getRobotX()) ? 1.0 : 0.0,

                // LEFT Direction
                state.getRobotX() == 0 || RobotLakeHelper.isLocationSafe(state.getLake(), state.getRobotY(), state.getRobotX() - 1) ? 1.0 : 0.0,
        };
    }

    private Map<String, Object> getChannelsData() {
        return new HashMap<String, Object>() {{
            put("tracker", getTrackerChannelData());
            put("radar", getRadarChannelData());
        }};
    }
}
