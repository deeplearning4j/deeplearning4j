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
package org.deeplearning4j.rl4j.observation.transform;

import lombok.Getter;
import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.rl4j.helper.INDArrayHelper;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.shade.guava.collect.Maps;
import org.datavec.api.transform.Operation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/**
 * A TransformProcess will build an {@link Observation Observation} from the raw data coming from the environment.
 * Three types of steps are available:
 *   1) A {@link FilterOperation FilterOperation}: Used to determine if an observation should be skipped.
 *   2) An {@link Operation Operation}: Applies a transform and/or conversion to an observation channel.
 *   3) A {@link DataSetPreProcessor DataSetPreProcessor}: Applies a DataSetPreProcessor to the observation channel. The data in the channel must be a DataSet.
 *
 * Instances of the three types above can be called in any order. The only requirement is that when build() is called,
 * all channels must be instances of INDArrays or DataSets
 *
 *   NOTE: Presently, only single-channels observations are supported.
 *
 * @author Alexandre Boulanger
 */
public class TransformProcess {

    private final List<Map.Entry<String, Object>> operations;
    @Getter
    private final String[] channelNames;
    private final HashSet<String> operationsChannelNames;

    private TransformProcess(Builder builder, String... channelNames) {
        operations = builder.operations;
        this.channelNames = channelNames;
        this.operationsChannelNames = builder.requiredChannelNames;
    }

    /**
     * This method will call reset() of all steps implementing {@link ResettableOperation ResettableOperation} in the transform process.
     */
    public void reset() {
        for(Map.Entry<String, Object> entry : operations) {
            if(entry.getValue() instanceof ResettableOperation) {
                ((ResettableOperation) entry.getValue()).reset();
            }
        }
    }

    /**
     * Transforms the channel data into an Observation or a skipped observation depending on the specific steps in the transform process.
     *
     * @param channelsData A Map that maps the channel name to its data.
     * @param currentObservationStep The observation's step number within the episode.
     * @param isFinalObservation True if the observation is the last of the episode.
     * @return An observation (may be a skipped observation)
     */
    public Observation transform(Map<String, Object> channelsData, int currentObservationStep, boolean isFinalObservation) {
        // null or empty channelData
        Preconditions.checkArgument(channelsData != null && channelsData.size() != 0, "Error: channelsData not supplied.");

        // Check that all channels have data
        for(Map.Entry<String, Object> channel : channelsData.entrySet()) {
            Preconditions.checkNotNull(channel.getValue(), "Error: data of channel '%s' is null", channel.getKey());
        }

        // Check that all required channels are present
        for(String channelName : operationsChannelNames) {
            Preconditions.checkArgument(channelsData.containsKey(channelName), "The channelsData map does not contain the channel '%s'", channelName);
        }

        for(Map.Entry<String, Object> entry : operations) {

            // Filter
            if(entry.getValue() instanceof FilterOperation) {
                FilterOperation filterOperation = (FilterOperation)entry.getValue();
                if(filterOperation.isSkipped(channelsData, currentObservationStep, isFinalObservation)) {
                    return Observation.SkippedObservation;
                }
            }

            // Transform
            // null results are considered skipped observations
            else if(entry.getValue() instanceof Operation) {
                Operation transformOperation = (Operation)entry.getValue();
                Object transformed = transformOperation.transform(channelsData.get(entry.getKey()));
                if(transformed == null) {
                    return Observation.SkippedObservation;
                }
                channelsData.replace(entry.getKey(), transformed);
            }

            // PreProcess
            else if(entry.getValue() instanceof DataSetPreProcessor) {
                Object channelData = channelsData.get(entry.getKey());
                DataSetPreProcessor dataSetPreProcessor = (DataSetPreProcessor)entry.getValue();
                if(!(channelData instanceof DataSet)) {
                    throw new IllegalArgumentException("The channel data must be a DataSet to call preProcess");
                }
                dataSetPreProcessor.preProcess((DataSet)channelData);
            }

            else {
                throw new IllegalArgumentException(String.format("Unknown operation: '%s'", entry.getValue().getClass().getName()));
            }
        }

        // Check that all channels used to build the observation are instances of
        // INDArray or DataSet
        // TODO: Add support for an interface with a toINDArray() method
        for(String channelName : channelNames) {
            Object channelData = channelsData.get(channelName);

            INDArray finalChannelData;
            if(channelData instanceof DataSet) {
                finalChannelData = ((DataSet)channelData).getFeatures();
            }
            else if(channelData instanceof INDArray) {
                finalChannelData = (INDArray) channelData;
            }
            else {
                throw new IllegalStateException("All channels used to build the observation must be instances of DataSet or INDArray");
            }

            // The dimension 0 of all INDArrays must be 1 (batch count)
            channelsData.replace(channelName, INDArrayHelper.forceCorrectShape(finalChannelData));
        }

        INDArray[] data = new INDArray[channelNames.length];
        for(int i = 0; i < channelNames.length; ++i) {
            data[i] = ((INDArray) channelsData.get(channelNames[i]));
        }

        return new Observation(data);
    }

    /**
     * @return An instance of a builder
     */
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

        private final List<Map.Entry<String, Object>> operations = new ArrayList<Map.Entry<String, Object>>();
        private final HashSet<String> requiredChannelNames = new HashSet<String>();

        /**
         * Add a filter to the transform process steps. Used to skip observations on certain conditions.
         * See {@link FilterOperation FilterOperation}
         * @param filterOperation An instance
         */
        public Builder filter(FilterOperation filterOperation) {
            Preconditions.checkNotNull(filterOperation, "The filterOperation must not be null");

            operations.add((Map.Entry)Maps.immutableEntry(null, filterOperation));
            return this;
        }

        /**
         * Add a transform to the steps. The transform can change the data and / or change the type of the data
         * (e.g. Byte[] to a ImageWritable)
         *
         * @param targetChannel The name of the channel to which the transform is applied.
         * @param transformOperation An instance of {@link Operation Operation}
         */
        public Builder transform(String targetChannel, Operation transformOperation) {
            Preconditions.checkNotNull(targetChannel, "The targetChannel must not be null");
            Preconditions.checkNotNull(transformOperation, "The transformOperation must not be null");

            requiredChannelNames.add(targetChannel);
            operations.add((Map.Entry)Maps.immutableEntry(targetChannel, transformOperation));
            return this;
        }

        /**
         * Add a DataSetPreProcessor to the steps. The channel must be a DataSet instance at this step.
         * @param targetChannel The name of the channel to which the pre processor is applied.
         * @param dataSetPreProcessor
         */
        public Builder preProcess(String targetChannel, DataSetPreProcessor dataSetPreProcessor) {
            Preconditions.checkNotNull(targetChannel, "The targetChannel must not be null");
            Preconditions.checkNotNull(dataSetPreProcessor, "The dataSetPreProcessor must not be null");

            requiredChannelNames.add(targetChannel);
            operations.add((Map.Entry)Maps.immutableEntry(targetChannel, dataSetPreProcessor));
            return this;
        }

        /**
         * Builds the TransformProcess.
         * @param channelNames A subset of channel names to be used to build the observation
         * @return An instance of TransformProcess
         */
        public TransformProcess build(String... channelNames) {
            if(channelNames.length == 0) {
                throw new IllegalArgumentException("At least one channel must be supplied.");
            }

            for(String channelName : channelNames) {
                Preconditions.checkNotNull(channelName, "Error: got a null channel name");
                requiredChannelNames.add(channelName);
            }

            return new TransformProcess(this, channelNames);
        }
    }
}
