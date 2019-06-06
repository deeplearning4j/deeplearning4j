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

package org.deeplearning4j.malmo;

import java.util.HashMap;

import org.json.JSONArray;
import org.json.JSONObject;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.microsoft.msr.malmo.TimestampedStringVector;
import com.microsoft.msr.malmo.WorldState;

/**
 * Observation space that contains a grid of Minecraft blocks
 * @author howard-abrams (howard.abrams@ca.com) on 1/12/17.
 */
public class MalmoObservationSpaceGrid extends MalmoObservationSpace {
    static final int MAX_BLOCK = 4095;

    int size;
    String name;

    int totalSize;

    HashMap<String, Integer> blockMap = new HashMap<String, Integer>();

    /**
     * Construct observation space from a array of blocks policy should distinguish between.
     * 
     * @param name Name given to Grid element in mission specification
     * @param xSize total x size of grid
     * @param ySize total y size of grid
     * @param zSize total z size of grid
     * @param blocks Array of block names to distinguish between. Supports combination of individual strings and/or arrays of strings to map multiple block types to a single observation value. If not specified, it will dynamically map block names to integers - however, because these will be mapped as they are seen, different missions may have different mappings!
     */
    public MalmoObservationSpaceGrid(String name, int xSize, int ySize, int zSize, Object... blocks) {
        this.name = name;

        this.totalSize = xSize * ySize * zSize;

        if (blocks.length == 0) {
            this.size = MAX_BLOCK;
        } else {
            this.size = blocks.length;

            // Mapping is 1-based;  0 == all other types
            for (int i = 0; i < blocks.length; ++i) {
                if (blocks[i] instanceof String)
                    blockMap.put((String) blocks[i], i + 1);
                else
                    for (String block : (String[]) blocks[i])
                        blockMap.put(block, i + 1);
            }
        }
    }

    @Override
    public String getName() {
        return "Box(" + totalSize + ")";
    }

    @Override
    public int[] getShape() {
        return new int[] {totalSize};
    }

    @Override
    public INDArray getLow() {
        INDArray low = Nd4j.create(getShape());
        return low;
    }

    @Override
    public INDArray getHigh() {
        INDArray high = Nd4j.linspace(255, 255, totalSize).reshape(getShape());
        return high;
    }

    public MalmoBox getObservation(WorldState world_state) {
        TimestampedStringVector observations = world_state.getObservations();

        if (observations.isEmpty())
            return null;

        String obs_text = observations.get((int) (observations.size() - 1)).getText();

        JSONObject observation = new JSONObject(obs_text);
        JSONArray blocks = observation.getJSONArray(name);

        double blockTypes[] = new double[totalSize];

        for (int i = 0; i < totalSize; ++i) {
            String block = blocks.getString(i);
            Integer mapped = blockMap.get(block);

            if (size == MAX_BLOCK && mapped == null) {
                mapped = blockMap.size();
                blockMap.put(block, mapped);
            }

            blockTypes[i] = mapped == null ? 0 : mapped;
        }

        return new MalmoBox(blockTypes);
    }
}
