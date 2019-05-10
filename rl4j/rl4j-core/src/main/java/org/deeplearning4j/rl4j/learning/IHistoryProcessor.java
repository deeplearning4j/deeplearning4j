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

package org.deeplearning4j.rl4j.learning;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Value;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/6/16.
 *
 * An IHistoryProcessor come directly from the atari DQN paper.
 * It applies pre-processing the pixels of one state (gray-scaling + resizing)
 * then stacks it in different channels to be fed to a conv net
 */
public interface IHistoryProcessor {

    Configuration getConf();

    /** Returns compressed arrays, which must be rescaled based
     *  on the value returned by {@link #getScale()}. */
    INDArray[] getHistory();

    void record(INDArray image);

    void add(INDArray image);

    void startMonitor(String filename, int[] shape);

    void stopMonitor();

    boolean isMonitoring();

    /** Returns the scale of the arrays returned by {@link #getHistory()}, typically 255. */
    double getScale();

    @AllArgsConstructor
    @Builder
    @Value
    public static class Configuration {
        int historyLength;
        int rescaledWidth;
        int rescaledHeight;
        int croppingWidth;
        int croppingHeight;
        int offsetX;
        int offsetY;
        int skipFrame;

        public Configuration() {
            historyLength = 4;
            rescaledWidth = 84;
            rescaledHeight = 84;
            croppingWidth = 84;
            croppingHeight = 84;
            offsetX = 0;
            offsetY = 0;
            skipFrame = 4;
        }

        public int[] getShape() {
            return new int[] {getHistoryLength(), getCroppingHeight(), getCroppingWidth()};
        }
    }
}
