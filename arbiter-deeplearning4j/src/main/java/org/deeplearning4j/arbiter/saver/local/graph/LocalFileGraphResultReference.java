/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
 *
 */
package org.deeplearning4j.arbiter.saver.local.graph;

import lombok.AllArgsConstructor;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.arbiter.GraphConfiguration;
import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

/**
 * Result reference for saving ComputationGraphs locally
 *
 * @param <A> Additional evaluation
 * @author Alex Black
 */
@AllArgsConstructor
public class LocalFileGraphResultReference<A> implements ResultReference<GraphConfiguration, ComputationGraph, A> {

    private int index;
    private String dir;
    private File configFile;
    private File networkParamsFile;
    private File scoreFile;
    private File additionalResultsFile;
    private File esConfigFile;
    private File numEpochsFile;
    private Candidate<GraphConfiguration> candidate;

    @Override
    public OptimizationResult<GraphConfiguration, ComputationGraph, A> getResult() throws IOException {
        String jsonConfig = FileUtils.readFileToString(configFile);
        INDArray params;
        try (DataInputStream dis = new DataInputStream(new FileInputStream(networkParamsFile))) {
            params = Nd4j.read(dis);
        }

        ComputationGraphConfiguration conf = ComputationGraphConfiguration.fromJson(jsonConfig);
        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        net.setParams(params);

        String scoreStr = FileUtils.readFileToString(scoreFile);
        //TODO: properly parsing. Probably want to store additional info other than just score...
        double d = Double.parseDouble(scoreStr);

        EarlyStoppingConfiguration<ComputationGraph> earlyStoppingConfiguration = null;
        if (esConfigFile != null && esConfigFile.exists()) {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(esConfigFile))) {
                earlyStoppingConfiguration = (EarlyStoppingConfiguration<ComputationGraph>) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException("Error loading early stopping configuration", e);
            }
        }
        int nEpochs = 1;
        if (numEpochsFile != null && numEpochsFile.exists()) {
            String numEpochs = FileUtils.readFileToString(numEpochsFile);
            nEpochs = Integer.parseInt(numEpochs);
        }

        GraphConfiguration dl4JConfiguration = new GraphConfiguration(conf, earlyStoppingConfiguration, nEpochs);


        A additionalResults;
        if (additionalResultsFile.exists()) {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(additionalResultsFile))) {
                additionalResults = (A) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException("Error loading additional results", e);
            }
        } else {
            additionalResults = null;
        }

        return new OptimizationResult<>(candidate, net, d, index, additionalResults);
    }

    @Override
    public String toString() {
        return "LocalFileGraphResultReference(" + dir + ")";
    }
}
