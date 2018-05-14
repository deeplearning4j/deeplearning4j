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
package org.deeplearning4j.arbiter.saver.local;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.arbiter.DL4JConfiguration;
import org.deeplearning4j.arbiter.GraphConfiguration;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.*;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Basic MultiLayerNetwork saver. Saves config, parameters and score to: baseDir/0/, baseDir/1/, etc
 * where index is given by OptimizationResult.getIndex()
 *
 * @author Alex Black
 */
@Slf4j
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode
public class FileModelSaver implements ResultSaver {
    @JsonProperty
    private String path;
    private File fPath;

    @JsonCreator
    public FileModelSaver(@NonNull String path) {
        this(new File(path));
    }

    public FileModelSaver(@NonNull File file){
        this.path = file.getPath();
        this.fPath = file;

        if(!fPath.exists()){
            fPath.mkdirs();
        } else if (!fPath.isDirectory()) {
            throw new IllegalArgumentException("Invalid path: exists and is not directory. " + path);
        }

        log.info("FileModelSaver saving networks to local directory: {}", path);
    }

    @Override
    public ResultReference saveModel(OptimizationResult result, Object modelResult) throws IOException {
        String dir = new File(path, result.getIndex() + "/").getAbsolutePath();

        File f = new File(dir);
        f.mkdir();

        File modelFile = new File(FilenameUtils.concat(dir, "model.bin"));
        File scoreFile = new File(FilenameUtils.concat(dir, "score.txt"));
        File additionalResultsFile = new File(FilenameUtils.concat(dir, "additionalResults.bin"));
        File esConfigFile = new File(FilenameUtils.concat(dir, "earlyStoppingConfig.bin"));
        File numEpochsFile = new File(FilenameUtils.concat(dir, "numEpochs.txt"));

        FileUtils.writeStringToFile(scoreFile, String.valueOf(result.getScore()));

        Model m = (Model) modelResult;
        ModelSerializer.writeModel(m, modelFile, true);


        Object additionalResults = result.getModelSpecificResults();
        if (additionalResults != null && additionalResults instanceof Serializable) {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(additionalResultsFile))) {
                oos.writeObject(additionalResults);
            }
        }

        //Write early stopping configuration (if present) to file:
        int nEpochs;
        EarlyStoppingConfiguration esc;
        if (result.getCandidate().getValue() instanceof DL4JConfiguration) {
            DL4JConfiguration c = ((DL4JConfiguration) result.getCandidate().getValue());
            esc = c.getEarlyStoppingConfiguration();
            nEpochs = c.getNumEpochs();
        } else {
            GraphConfiguration c = ((GraphConfiguration) result.getCandidate().getValue());
            esc = c.getEarlyStoppingConfiguration();
            nEpochs = c.getNumEpochs();
        }


        if (esc != null) {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(esConfigFile))) {
                oos.writeObject(esc);
            }
        } else {
            FileUtils.writeStringToFile(numEpochsFile, String.valueOf(nEpochs));
        }

        log.debug("Deeplearning4j model result (id={}, score={}) saved to directory: {}", result.getIndex(),
                        result.getScore(), dir);

        boolean isGraph = m instanceof ComputationGraph;
        return new LocalFileNetResultReference(result.getIndex(), dir, isGraph, modelFile, scoreFile,
                        additionalResultsFile, esConfigFile, numEpochsFile, result.getCandidate());
    }

    @Override
    public List<Class<?>> getSupportedCandidateTypes() {
        return Collections.<Class<?>>singletonList(Object.class);
    }

    @Override
    public List<Class<?>> getSupportedModelTypes() {
        return Arrays.<Class<?>>asList(MultiLayerNetwork.class, ComputationGraph.class);
    }

    @Override
    public String toString() {
        return "FileModelSaver(path=" + path + ")";
    }
}
