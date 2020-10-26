/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.mdp.robotlake;

import org.nd4j.linalg.api.rng.Random;

public class RobotLakeMap {
    private static final double SAFE_ICE_PROBABILITY = 0.8;

    private final Random rnd;

    private final char[][] lake;
    public final int size;

    public RobotLakeMap(int size, Random rnd) {
        this.size = size;
        this.rnd = rnd;
        lake = new char[size][size];
    }

    public void generateLake(int playerY, int playerX, int goalY, int goalX) {
        for(int y = 0; y < size; ++y) {
            for(int x = 0; x < size; ++x) {
                lake[y][x] = rnd.nextDouble() <= SAFE_ICE_PROBABILITY
                        ? RobotLake.ICE
                        : RobotLake.HOLE;
            }
        }

        lake[goalY][goalX] = RobotLake.GOAL;
        lake[playerY][playerX] = RobotLake.ICE;
    }

    public char getLocation(int y, int x) {
        return lake[y][x];
    }
}
