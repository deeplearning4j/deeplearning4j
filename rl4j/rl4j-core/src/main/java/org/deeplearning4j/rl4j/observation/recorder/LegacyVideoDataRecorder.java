/*******************************************************************************
 * Copyright (c) 2020 Skymind, Inc.
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

package org.deeplearning4j.rl4j.observation.recorder;

import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.observation.channel.ChannelData;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * A {@link DataRecorder DataRecorder} that will record the video channel of legacy MDPs (those using Encodable)
 *
 * @author Alexandre Boulanger
 */
public class LegacyVideoDataRecorder implements DataRecorder {

    private final IHistoryProcessor historyProcessor;

    /**
     * Creates an instance of LegacyVideoDataRecorder
     * @param historyProcessor
     */
    public LegacyVideoDataRecorder(IHistoryProcessor historyProcessor) {
        // TODO: Use VideoRecorder instead of HistoryProcessor
        this.historyProcessor = historyProcessor;
    }

    /**
     * Records an observation into a video frame.
     * @param channelDataList currently uses the first channel as the video channel
     */
    @Override
    public void record(List<ChannelData> channelDataList) {
        INDArray features = channelDataList.get(0).toINDArray();
        historyProcessor.record(features);
    }
}
