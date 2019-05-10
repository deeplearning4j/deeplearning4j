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

package org.deeplearning4j.arbiter.saver.local;

import lombok.AllArgsConstructor;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.arbiter.DL4JConfiguration;
import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

/**
 * Result reference for MultiLayerNetworks and ComputationGraphs saved to local file system
 */
@AllArgsConstructor
public class LocalFileNetResultReference implements ResultReference {

    private int index;
    private String dir;
    private boolean isGraph;
    private File modelFile;
    private File scoreFile;
    private File additionalResultsFile;
    private File esConfigFile;
    private File numEpochsFile;
    private Candidate<DL4JConfiguration> candidate;

    @Override
    public OptimizationResult getResult() throws IOException {


        String scoreStr = FileUtils.readFileToString(scoreFile);
        //TODO: properly parsing. Probably want to store additional info other than just score...
        double d = Double.parseDouble(scoreStr);

        EarlyStoppingConfiguration earlyStoppingConfiguration = null;
        if (esConfigFile != null && esConfigFile.exists()) {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(esConfigFile))) {
                earlyStoppingConfiguration = (EarlyStoppingConfiguration) ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException("Error loading early stopping configuration", e);
            }
        }
        int nEpochs = 1;
        if (numEpochsFile != null && numEpochsFile.exists()) {
            String numEpochs = FileUtils.readFileToString(numEpochsFile);
            nEpochs = Integer.parseInt(numEpochs);
        }



        Object additionalResults;
        if (additionalResultsFile.exists()) {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(additionalResultsFile))) {
                additionalResults = ois.readObject();
            } catch (ClassNotFoundException e) {
                throw new RuntimeException("Error loading additional results", e);
            }
        } else {
            additionalResults = null;
        }

        return new OptimizationResult(candidate, d, index, additionalResults, null, this);
    }

    @Override
    public Object getResultModel() throws IOException {
        Model m;
        if (isGraph) {
            m = ModelSerializer.restoreComputationGraph(modelFile, false);
        } else {
            m = ModelSerializer.restoreMultiLayerNetwork(modelFile, false);
        }
        return m;
    }

    @Override
    public String toString() {
        return "LocalFileNetResultReference(" + dir + ")";
    }
}
