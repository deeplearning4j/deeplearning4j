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

package org.deeplearning4j.optimize.listeners;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * CollectScoresIterationListener simply stores the model scores internally (along with the iteration) every 1 or N
 * iterations (this is configurable). These scores can then be obtained or exported.
 *
 * @author Alex Black
 */
public class CollectScoresIterationListener extends BaseTrainingListener {

    private int frequency;
    private int iterationCount = 0;
    private List<Pair<Integer, Double>> scoreVsIter = new ArrayList<>();

    /**
     * Constructor for collecting scores with default saving frequency of 1
     */
    public CollectScoresIterationListener() {
        this(1);
    }

    /**
     * Constructor for collecting scores with the specified frequency.
     * @param frequency    Frequency with which to collect/save scores
     */
    public CollectScoresIterationListener(int frequency) {
        if (frequency <= 0)
            frequency = 1;
        this.frequency = frequency;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if (++iterationCount % frequency == 0) {
            double score = model.score();
            scoreVsIter.add(new Pair<>(iterationCount, score));
        }
    }

    public List<Pair<Integer, Double>> getScoreVsIter() {
        return scoreVsIter;
    }

    /**
     * Export the scores in tab-delimited (one per line) UTF-8 format.
     */
    public void exportScores(OutputStream outputStream) throws IOException {
        exportScores(outputStream, "\t");
    }

    /**
     * Export the scores in delimited (one per line) UTF-8 format with the specified delimiter
     *
     * @param outputStream Stream to write to
     * @param delimiter    Delimiter to use
     */
    public void exportScores(OutputStream outputStream, String delimiter) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("Iteration").append(delimiter).append("Score");
        for (Pair<Integer, Double> p : scoreVsIter) {
            sb.append("\n").append(p.getFirst()).append(delimiter).append(p.getSecond());
        }
        outputStream.write(sb.toString().getBytes("UTF-8"));
    }

    /**
     * Export the scores to the specified file in delimited (one per line) UTF-8 format, tab delimited
     *
     * @param file File to write to
     */
    public void exportScores(File file) throws IOException {
        exportScores(file, "\t");
    }

    /**
     * Export the scores to the specified file in delimited (one per line) UTF-8 format, using the specified delimiter
     *
     * @param file      File to write to
     * @param delimiter Delimiter to use for writing scores
     */
    public void exportScores(File file, String delimiter) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(file)) {
            exportScores(fos, delimiter);
        }
    }

}
