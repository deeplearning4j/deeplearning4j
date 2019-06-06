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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.microsoft.msr.malmo.ByteVector;
import com.microsoft.msr.malmo.TimestampedVideoFrameVector;
import com.microsoft.msr.malmo.WorldState;

/**
 * Observation space that contains a bitmap of Minecraft pixels
 * @author howard-abrams (howard.abrams@ca.com) on 2/19/17.
 */
public class MalmoObservationSpacePixels extends MalmoObservationSpace {
    int xSize;
    int ySize;

    HashMap<String, Integer> blockMap = new HashMap<String, Integer>();

    /**
     * Construct observation space from a bitmap size. Assumes 3 color channels.
     * 
     * @param xSize total x size of bitmap
     * @param ySize total y size of bitmap
     */
    public MalmoObservationSpacePixels(int xSize, int ySize) {
        this.xSize = xSize;
        this.ySize = ySize;
    }

    @Override
    public String getName() {
        return "Box(" + ySize + "," + xSize + ",3)";
    }

    @Override
    public int[] getShape() {
        return new int[] {ySize, xSize, 3};
    }

    @Override
    public INDArray getLow() {
        INDArray low = Nd4j.create(getShape());
        return low;
    }

    @Override
    public INDArray getHigh() {
        INDArray high = Nd4j.linspace(255, 255, xSize * ySize * 3).reshape(getShape());
        return high;
    }

    public MalmoBox getObservation(WorldState world_state) {
        TimestampedVideoFrameVector observations = world_state.getVideoFrames();

        double rawPixels[] = new double[xSize * ySize * 3];

        if (!observations.isEmpty()) {
            ByteVector pixels = observations.get((int) (observations.size() - 1)).getPixels();

            int i = 0;
            for (int x = 0; x < xSize; ++x)
                for (int y = 0; y < ySize; ++y)
                    for (int c = 2; c >= 0; --c) // BGR -> RGB
                    {
                        rawPixels[i] = pixels.get(3 * x * ySize + y * 3 + c) / 255.0;
                        i++;
                    }
        }

        return new MalmoBox(rawPixels);
    }
}
