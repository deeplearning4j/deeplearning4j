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
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.arbiter.GraphConfiguration;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


import java.io.*;
import java.nio.file.Files;

/**
 * Basic ComputationGraph saver. Saves config, parameters and score to: baseDir/0/, baseDir/1/, etc
 * where index is given by OptimizationResult.getIndex()
 *
 * @author Alex Black
 */
@Slf4j
@AllArgsConstructor
@NoArgsConstructor
public class LocalComputationGraphSaver<A> implements ResultSaver<GraphConfiguration, ComputationGraph, A> {
    private String path;
    private File fPath;

    public LocalComputationGraphSaver(String path) {
        if (path == null) throw new NullPointerException();
        this.path = path;
        this.fPath = new File(path);

        File baseDirectory = new File(path);
        if (!baseDirectory.isDirectory()) {
            throw new IllegalArgumentException("Invalid path: is not directory. " + path);
        }

        log.info("LocalComputationGraphSaver saving networks to local directory: {}", path);
    }

    @Override
    public ResultReference<GraphConfiguration, ComputationGraph, A> saveModel(OptimizationResult<GraphConfiguration, ComputationGraph, A> result) throws IOException {
        String dir = new File(path, result.getIndex() + "/").getAbsolutePath();

        File f = new File(dir);
        f.mkdir();

        File paramsFile = new File(FilenameUtils.concat(dir, "params.bin"));
        File jsonFile = new File(FilenameUtils.concat(dir, "config.json"));
        File scoreFile = new File(FilenameUtils.concat(dir, "score.txt"));
        File additionalResultsFile = new File(FilenameUtils.concat(dir, "additionalResults.bin"));
        File esConfigFile = new File(FilenameUtils.concat(dir, "earlyStoppingConfig.bin"));
        File numEpochsFile = new File(FilenameUtils.concat(dir, "numEpochs.txt"));

        FileUtils.writeStringToFile(scoreFile, String.valueOf(result.getScore()));
        String jsonConfig = result.getCandidate().getValue().getConfiguration().toJson();
        FileUtils.writeStringToFile(jsonFile, jsonConfig);


        if (result.getResult() != null) {
            INDArray params = result.getResult().params();
            try (DataOutputStream dos = new DataOutputStream(Files.newOutputStream(paramsFile.toPath()))) {
                Nd4j.write(params, dos);
            }
        }


        A additionalResults = result.getModelSpecificResults();
        if (additionalResults != null) {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(additionalResultsFile))) {
                oos.writeObject(additionalResults);
            }
        }

        //Write early stopping configuration (if present) to file:
        EarlyStoppingConfiguration esc = result.getCandidate().getValue().getEarlyStoppingConfiguration();
        if (esc != null) {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(esConfigFile))) {
                oos.writeObject(esc);
            }
        } else {
            int nEpochs = result.getCandidate().getValue().getNumEpochs();
            FileUtils.writeStringToFile(numEpochsFile, String.valueOf(nEpochs));
        }

        log.debug("Deeplearning4j model result (id={}, score={}) saved to directory: {}", result.getIndex(), result.getScore(), dir);

        return new LocalFileGraphResultReference<>(result.getIndex(), dir,
                jsonFile,
                paramsFile,
                scoreFile,
                additionalResultsFile,
                esConfigFile,
                numEpochsFile,
                result.getCandidate());
    }

    @Override
    public String toString() {
        return "LocalMultiLayerNetworkScoreSaver(path=" + fPath.getAbsolutePath() + ")";
    }
}
