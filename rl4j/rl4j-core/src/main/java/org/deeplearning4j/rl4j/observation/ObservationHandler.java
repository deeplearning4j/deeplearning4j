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

package org.deeplearning4j.rl4j.observation;

import lombok.Setter;
import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.observation.channel.ChannelData;
import org.deeplearning4j.rl4j.observation.prefiltering.PreFilter;
import org.deeplearning4j.rl4j.observation.recorder.DataRecorder;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * A handler that can create and record {@link Observation Observations} from raw environment data.
 *
 * When building, the observation should contain exactly one data channel (presently only single-channel observations are supported)
 * The observation is built this way:
 * <ol>
 *     <li>The channel data is recorded (if at least one {@link DataRecorder DataRecorder} has been added)</li>
 *     <li>The observation is filtered (if at least one {@link PreFilter PreFilter}  has been added). The observation must pass <b>all</b> filters, otherwise it is skipped</li>
 *     <li>The observation is pre-processed (if a {@link DataSetPreProcessor DataSetPreProcessor} has been set)</li>
 *     <li>The observation is assembled with past observation if a IHistoryProcessor has been set (this is subject to change in the future)</li>
 * </ol>
 *
 * @author Alexandre Boulanger
 */
public class ObservationHandler {

    private int currentEpisodeStep;

    /**
     * A {@link DataSetPreProcessor DataSetPreProcessor} that will be applied to non-skipped observations.
     * {@link org.nd4j.linalg.dataset.api.preprocessor.CompositeDataSetPreProcessor CompositeDataSetPreProcessor} can be used to apply multiple processors
     */
    @Setter
    private DataSetPreProcessor dataSetPreProcessor;

    private final List<PreFilter> preFilters = new ArrayList<PreFilter>();
    private final List<DataRecorder> dataRecorders = new ArrayList<DataRecorder>();

    @Setter
    private IHistoryProcessor historyProcessor; // FIXME: to be removed eventually

    /**
     * Starts the creation of an {@link Observation Observation} in a fluent way
     * @return an {@link ObservationBuilder ObservationBuilder}
     */
    public ObservationBuilder newObservation() {
        return new ObservationBuilder(this, currentEpisodeStep++);
    }

    /**
     * Add a {@link PreFilter PreFilter} to the pre-filters list. PreFilters are invoked in the same order that they are added.
     * @param preFilter a PreFilter
     */
    public void addPreFilter(PreFilter preFilter) {
        Preconditions.checkNotNull(preFilter, "preFilter should not be null");
        preFilters.add(preFilter);
    }

    /**
     * Add a {@link DataRecorder DataRecorder} to the recorders list.
     * @param dataRecorder a DataRecorder
     */
    public void addDataRecorder(DataRecorder dataRecorder) {
        Preconditions.checkNotNull(dataRecorder, "dataRecorder should not be null");
        dataRecorders.add(dataRecorder);
    }

    /**
     * Resets the ObservationHandler's state to how it should be at the start of an episode.
     */
    public void reset() {
        currentEpisodeStep = 0;
        if(historyProcessor != null) {
            historyProcessor.reset();
        }
    }

    private Observation buildObservation(List<ChannelData> channelDataList, int observationEpisodeStep, boolean isFinalObservation) {
        // Record
        recordData(channelDataList);

        // Filter
        boolean isPassing = preFilterObservation(channelDataList, observationEpisodeStep, isFinalObservation);
        if(!isPassing) {
            return Observation.SkippedObservation;
        }

        DataSet dataSet = createDataSet(channelDataList);

        // Pre-Process
        DataSetPreProcessor preProcessor = dataSetPreProcessor;
        if(preProcessor != null) {
            preProcessor.preProcess(dataSet);
        }

        // Assemble (currently only necessary with HistoryProcessor)
        // TODO: refac history stacking
        if(historyProcessor != null) {
            historyProcessor.add(dataSet.getFeatures());

            if(!historyProcessor.isHistoryReady()) {
                return Observation.SkippedObservation;
            }
            INDArray[] frames = historyProcessor.getHistory();
            INDArray stackedFrames = Nd4j.concat(0, frames);
            dataSet.setFeatures(stackedFrames);

            enforceObservationShape(dataSet);
        }

        return new Observation(dataSet);
    }

    private DataSet createDataSet(List<ChannelData> channelDataList) {
        // FIXME: Only single channel observation supported right now.
        INDArray features = channelDataList.get(0).toINDArray();
        DataSet dataSet = new org.nd4j.linalg.dataset.DataSet(features, null);
        enforceObservationShape(dataSet);

        return dataSet;
    }

    private boolean preFilterObservation(List<ChannelData> channelDataList, int observationEpisodeStep, boolean isFinalObservation) {
        boolean isPassing = true;
        Iterator<PreFilter> preFilterIterator = preFilters.iterator();
        while (preFilterIterator.hasNext() && isPassing) {
            isPassing &= preFilterIterator.next().isPassing(channelDataList, observationEpisodeStep, isFinalObservation);
        }
        return isPassing;
    }

    private void recordData(List<ChannelData> channelDataList) {
        for(DataRecorder recorder : dataRecorders) {
            recorder.record(channelDataList);
        }
    }

    private void enforceObservationShape(DataSet dataSet) {
        INDArray features = dataSet.getFeatures();
        long[] shape = features.shape();
        if(shape[0] != 1) {
            long[] nshape = new long[shape.length + 1];
            nshape[0] = 1;
            System.arraycopy(shape, 0, nshape, 1, shape.length);

            dataSet.setFeatures(features.reshape(nshape));
        }
    }

    public static class ObservationBuilder {
        private final ObservationHandler observationHandler;
        private final int observationEpisodeStep;
        private List<ChannelData> channelDataList = new ArrayList<ChannelData>();
        private boolean isFinalObservation;

        public ObservationBuilder(ObservationHandler observationHandler, int observationEpisodeStep) {
            this.observationHandler = observationHandler;
            this.observationEpisodeStep = observationEpisodeStep;
        }

        /**
         * Indicates whether the observation is the last of the episode
         * @param value true if it's the last observation
         * @return
         */
        public ObservationBuilder isFinalObservation(boolean value) {
            isFinalObservation = value;
            return this;
        }

        /**
         * Mandatory. Adds a data channel to the observation
         * Note: Presently, only single-channel observations are supported.
         *
         * @param channelData a {@link ChannelData ChannelData} instance.
         * @return
         */
        public ObservationBuilder addChannelData(ChannelData channelData) {
            if(!channelDataList.isEmpty()) {
                throw new NotImplementedException("More than one data channel is not yet implemented.");
            }

            channelDataList.add(channelData);
            return this;
        }

        /**
         * Builds the {@link Observation}
         * @return an Observation instance
         */
        public Observation build() {
            Preconditions.checkArgument(!channelDataList.isEmpty(), "Channel data not supplied");

            return observationHandler.buildObservation(channelDataList, observationEpisodeStep, isFinalObservation);
        }


    }

}
