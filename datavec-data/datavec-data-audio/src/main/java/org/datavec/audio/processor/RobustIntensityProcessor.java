/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

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
