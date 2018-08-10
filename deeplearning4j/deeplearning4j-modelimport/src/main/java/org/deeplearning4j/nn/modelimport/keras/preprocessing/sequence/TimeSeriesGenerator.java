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

package org.deeplearning4j.nn.modelimport.keras.preprocessing.sequence;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import lombok.Data;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.preprocessing.text.KerasTokenizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils.parseJsonString;

/**
 * Java port of Keras' TimeSeriesGenerator, see https://keras.io/preprocessing/sequence/
 *
 * @author Max Pumperla
 */
@Data
public class TimeSeriesGenerator {

    private final static int DEFAULT_SAMPLING_RATE = 1;
    private final static int DEFAULT_STRIDE = 1;
    private final static Integer DEFAULT_START_INDEX = 0;
    private final static Integer DEFAULT_END_INDEX = null;
    private final static boolean DEFAULT_SHUFFLE = false;
    private final static boolean DEFAULT_REVERSE = false;
    private final static int DEFAULT_BATCH_SIZE = 128;

    private INDArray data;
    private INDArray targets;
    private int length;
    private int samplingRate;
    private int stride;
    private int startIndex;
    private int endIndex;
    private boolean shuffle;
    private boolean reverse;
    private int batchSize;

    // TODO: pad_sequences, make_sampling_table, skipgrams util?

    public static TimeSeriesGenerator fromJson(String jsonFileName)
            throws IOException, InvalidKerasConfigurationException {
        String json = new String(Files.readAllBytes(Paths.get(jsonFileName)));
        Map<String, Object> timeSeriesBaseConfig = parseJsonString(json);
        Map<String, Object> timeSeriesConfig;
        if (timeSeriesBaseConfig.containsKey("config"))
            timeSeriesConfig = (Map<String, Object>) timeSeriesBaseConfig.get("config");
        else
            throw new InvalidKerasConfigurationException("No configuration found for Keras tokenizer");


        int length = (int) timeSeriesConfig.get("length");
        int samplingRate = (int) timeSeriesConfig.get("sampling_rate");
        int stride = (int) timeSeriesConfig.get("stride");
        int startIndex = (int) timeSeriesConfig.get("start_index");
        int endIndex = (int) timeSeriesConfig.get("end_index");
        int batchSize = (int) timeSeriesConfig.get("batch_size");

        boolean shuffle = (boolean) timeSeriesConfig.get("shuffle");
        boolean reverse = (boolean) timeSeriesConfig.get("reverse");


        Gson gson = new Gson();
        List<List<Double>> dataList = gson.fromJson((String) timeSeriesConfig.get("data"), new TypeToken<List<List<Double>>>() {}.getType());
        List<List<Double>> targetsList = gson.fromJson((String) timeSeriesConfig.get("targets"), new TypeToken<List<List<Double>>>() {}.getType());

        int dataPoints = dataList.size();
        int dataPointsPerRow = dataList.get(0).size();


        INDArray data = Nd4j.create(dataPoints, dataPointsPerRow);
        INDArray targets = Nd4j.create(dataPoints, dataPointsPerRow);
        for (int i = 0; i < dataPoints; i ++) {
            data.put(i, Nd4j.create(dataList.get(i)));
            targets.put(i, Nd4j.create(targetsList.get(i)));
        }


        TimeSeriesGenerator gen = new TimeSeriesGenerator(data, targets, length,
                samplingRate, stride, startIndex, endIndex, shuffle, reverse, batchSize);

        return gen;
    }
    public TimeSeriesGenerator(INDArray data, INDArray targets, int length, int samplingRate, int stride,
                               Integer startIndex, Integer endIndex, boolean shuffle, boolean reverse,
                               int batchSize) throws InvalidKerasConfigurationException {


        this.data = data;
        this.targets = targets;
        this.length = length;
        this.samplingRate = samplingRate;
        if (stride != 1)
            throw new InvalidKerasConfigurationException("currently no strides > 1 supported, got: " + stride);
        this.stride = stride;
        this.startIndex = startIndex + length;
        if (endIndex == null)
            endIndex = data.rows() -1;
        this.endIndex = endIndex;
        this.shuffle = shuffle;
        this.reverse = reverse;
        this.batchSize = batchSize;

        if (this.startIndex > this.endIndex)
            throw new IllegalArgumentException("Start index of sequence has to be smaller then end index, got " +
                    "startIndex : " + this.startIndex + " and endIndex: " + this.endIndex);
    }

    public TimeSeriesGenerator(INDArray data, INDArray targets, int length) throws InvalidKerasConfigurationException {
        this(data, targets, length, DEFAULT_SAMPLING_RATE, DEFAULT_STRIDE, DEFAULT_START_INDEX, DEFAULT_END_INDEX,
                DEFAULT_SHUFFLE, DEFAULT_REVERSE, DEFAULT_BATCH_SIZE);
    }

    public int length() {
        return (endIndex - startIndex + batchSize * stride) / (batchSize * stride);
    }

    public Pair<INDArray, INDArray> next(int index) {
        INDArray rows;
        if (shuffle) {
            rows = Nd4j.getRandom().nextInt(endIndex, new int[] {batchSize});
            rows.addi(startIndex);
        } else {
            int i = startIndex + batchSize + stride * index;
            // TODO: add stride arg to arange
            rows = Nd4j.arange(i, Math.min(i + batchSize * stride, endIndex + 1));
        }
        INDArray samples = Nd4j.create(rows.length(), length / samplingRate, data.columns());
        INDArray targets = Nd4j.create(rows.length(), this.targets.columns());

        for (int j = 0; j < rows.rows(); j++) {
            long idx = (long) rows.getDouble(j);
            INDArrayIndex indices = NDArrayIndex.interval(idx - this.length, this.samplingRate, idx);
            samples.putSlice(j, this.data.get(indices));
            INDArrayIndex point = NDArrayIndex.point((long) rows.getDouble(j));
            targets.putSlice(j, this.targets.get(point));
        }
        if (reverse)
            samples = Nd4j.reverse(samples);

        return new Pair<>(samples, targets);
    }
}

