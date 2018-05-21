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

import java.util.LinkedList;
import java.util.List;


public class TopManyPointsProcessorChain {

    private double[][] intensities;
    List<IntensityProcessor> processorList = new LinkedList<>();

    public TopManyPointsProcessorChain(double[][] intensities, int numPoints) {
        this.intensities = intensities;
        RobustIntensityProcessor robustProcessor = new RobustIntensityProcessor(intensities, numPoints);
        processorList.add(robustProcessor);
        process();
    }

    private void process() {
        for (IntensityProcessor processor : processorList) {
            processor.execute();
            intensities = processor.getIntensities();
        }
    }

    public double[][] getIntensities() {
        return intensities;
    }
}
