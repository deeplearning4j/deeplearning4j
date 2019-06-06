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

package org.datavec.audio.processor;


public class RobustIntensityProcessor implements IntensityProcessor {

    private double[][] intensities;
    private int numPointsPerFrame;

    public RobustIntensityProcessor(double[][] intensities, int numPointsPerFrame) {
        this.intensities = intensities;
        this.numPointsPerFrame = numPointsPerFrame;
    }

    public void execute() {

        int numX = intensities.length;
        int numY = intensities[0].length;
        double[][] processedIntensities = new double[numX][numY];

        for (int i = 0; i < numX; i++) {
            double[] tmpArray = new double[numY];
            System.arraycopy(intensities[i], 0, tmpArray, 0, numY);

            // pass value is the last some elements in sorted array	
            ArrayRankDouble arrayRankDouble = new ArrayRankDouble();
            double passValue = arrayRankDouble.getNthOrderedValue(tmpArray, numPointsPerFrame, false);

            // only passed elements will be assigned a value
            for (int j = 0; j < numY; j++) {
                if (intensities[i][j] >= passValue) {
                    processedIntensities[i][j] = intensities[i][j];
                }
            }
        }
        intensities = processedIntensities;
    }

    public double[][] getIntensities() {
        return intensities;
    }
}
