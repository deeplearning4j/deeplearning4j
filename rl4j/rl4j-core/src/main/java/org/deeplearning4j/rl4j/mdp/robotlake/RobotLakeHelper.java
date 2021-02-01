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
package org.deeplearning4j.rl4j.mdp.robotlake;

public class RobotLakeHelper {
    private static final byte SAFE_PATH_TO_LOCATION_EXISTS = (byte)1;
    private static final byte DANGEROUS_LOCATION = (byte)-1;
    private static final byte UNVISITED_LOCATION = (byte)0;

    public static boolean isGoalAtLocation(RobotLakeMap lake, int y, int x) {
        return lake.getLocation(y, x) == RobotLake.GOAL;
    }

    public static boolean pathExistsToGoal(RobotLakeMap lake, int startY, int startX) {
        byte[][] path = new byte[lake.size][lake.size];

        for (int y = 0; y < lake.size; ++y) {
            for (int x = 0; x < lake.size; ++x) {
                if(!isLocationSafe(lake, y, x)) {
                    path[y][x] = DANGEROUS_LOCATION;
                }
            }
        }

        path[startY][startX] = 1;
        int previousNumberOfLocations = 0;
        while (true) {
            int numberOfLocations = 0;

            for (int y = 0; y < lake.size; ++y) {
                for (int x = 0; x < lake.size; ++x) {
                    if (path[y][x] == SAFE_PATH_TO_LOCATION_EXISTS) {
                        ++numberOfLocations;
                        boolean hasFoundValidPath = updatePathSafetyAtLocation(lake, path, y - 1, x)
                                || updatePathSafetyAtLocation(lake, path, y, x - 1)
                                || updatePathSafetyAtLocation(lake, path, y + 1, x)
                                || updatePathSafetyAtLocation(lake, path, y, x + 1);

                        if(hasFoundValidPath) {
                            return true;
                        }
                    }
                }
            }

            if (previousNumberOfLocations == numberOfLocations) {
                return false;
            }
            previousNumberOfLocations = numberOfLocations;
        }
    }

    // returns true if goal has been reached
    private static boolean updatePathSafetyAtLocation(RobotLakeMap lake, byte[][] path, int y, int x) {
        if (y < 0 || y >= path.length || x < 0 || x >= path.length || path[y][x] != UNVISITED_LOCATION) {
            return false;
        }

        if(isGoalAtLocation(lake, y, x)) {
            return true;
        }

        path[y][x] = isLocationSafe(lake, y, x) ? SAFE_PATH_TO_LOCATION_EXISTS : DANGEROUS_LOCATION;

        return false;
    }

    public static boolean isLocationSafe(RobotLakeMap lake, int y, int x) {
        char contentOfLocation = lake.getLocation(y, x);
        return contentOfLocation == RobotLake.ICE
                || contentOfLocation == RobotLake.GOAL;
    }

}