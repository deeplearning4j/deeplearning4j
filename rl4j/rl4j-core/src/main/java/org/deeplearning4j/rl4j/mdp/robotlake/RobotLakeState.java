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
import org.nd4j.linalg.api.rng.Random;

public class RobotLakeState {
    private final int size;
    private final boolean areStartingPositionsRandom;
    private final Random rnd;

    @Getter
    private final RobotLakeMap lake;

    @Getter
    private int robotY, robotX;

    @Getter
    private int goalY, goalX;

    public RobotLakeState(int size, boolean areStartingPositionsRandom, Random rnd) {
        this.size = size;
        this.areStartingPositionsRandom = areStartingPositionsRandom;
        this.rnd = rnd;
        lake = new RobotLakeMap(size, rnd);
    }

    public void reset() {
        setRobotAndGoalLocations();
        generateValidPond();
    }

    private void generateValidPond() {
        int attempts = 0;
        while (attempts++ < 1000) {
            lake.generateLake(robotY, robotX, goalY, goalX);
            if(RobotLakeHelper.pathExistsToGoal(lake, robotY, robotX)) {
                return;
            }
        }

        throw new RuntimeException("Failed to generate a valid pond after 1000 attempts");
    }

    public void moveRobotLeft() {
        if(robotX > 0) {
            --robotX;
        }
    }

    public void moveRobotRight() {
        if(robotX < size - 1) {
            ++robotX;
        }
    }

    public void moveRobotUp() {
        if(robotY > 0) {
            --robotY;
        }
    }

    public void moveRobotDown() {
        if(robotY < size - 1) {
            ++robotY;
        }
    }

    private void setRobotAndGoalLocations() {
        if(areStartingPositionsRandom) {
            if (rnd.nextBoolean()) {
                // Robot on top side, goal on bottom side
                robotX = rnd.nextInt(size);
                robotY = 0;
                goalX = rnd.nextInt(size);
                goalY = size - 1;
            } else {
                // Robot on left side, goal on right side
                robotX = 0;
                robotY = rnd.nextInt(size);
                goalX = size - 1;
                goalY = rnd.nextInt(size);
            }
        } else {
            robotX = 0;
            robotY = 0;
            goalX = size - 1;
            goalY = size - 1;
        }
    }
}