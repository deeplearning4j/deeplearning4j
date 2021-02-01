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

package org.deeplearning4j.optimize.listeners;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
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
    //private List<Pair<Integer, Double>> scoreVsIter = new ArrayList<>();

    public static class ScoreStat {
        public static final int BUCKET_LENGTH = 10000;

        private int position = 0;
        private int bucketNumber = 1;
        private List<long[]> indexes;
        private List<double[]> scores;

        public ScoreStat() {
            indexes = new ArrayList<>(1);
            indexes.add(new long[BUCKET_LENGTH]);
            scores = new ArrayList<>(1);
            scores.add(new double[BUCKET_LENGTH]);
        }

        public List<long[]> getIndexes() {
            return indexes;
        }

        public List<double[]> getScores() {
            return scores;
        }

        public long[] getEffectiveIndexes() {
            return Arrays.copyOfRange(indexes.get(0), 0, position);
        }

        public double[] getEffectiveScores() {
            return Arrays.copyOfRange(scores.get(0), 0, position);
        }


        /*
            Originally scores array is initialized with BUCKET_LENGTH size.
            When data doesn't fit there - arrays size is increased for BUCKET_LENGTH,
            old data is copied and bucketNumber (counter of reallocations) being incremented.

            If we got more score points than MAX_VALUE - they are put to another item of scores list.
         */
        private void reallocateGuard() {
            if (position >= BUCKET_LENGTH * bucketNumber) {

                long fullLength = (long)BUCKET_LENGTH * bucketNumber;

                if (position == Integer.MAX_VALUE || fullLength >= Integer.MAX_VALUE) {
                    position = 0;
                    long[] newIndexes = new long[BUCKET_LENGTH];
                    double[] newScores = new double[BUCKET_LENGTH];
                    indexes.add(newIndexes);
                    scores.add(newScores);
                }
                else {
                    long[] newIndexes = new long[(int)fullLength + BUCKET_LENGTH];
                    double[] newScores = new double[(int)fullLength + BUCKET_LENGTH];
                    System.arraycopy(indexes.get(indexes.size()-1), 0, newIndexes, 0, (int)fullLength);
                    System.arraycopy(scores.get(scores.size()-1), 0, newScores, 0, (int)fullLength);
                    scores.remove(scores.size()-1);
                    indexes.remove(indexes.size()-1);
                    int lastIndex = scores.size() == 0 ? 0 : scores.size()-1;
                    scores.add(lastIndex, newScores);
                    indexes.add(lastIndex, newIndexes);
                }
                bucketNumber += 1;
            }
        }

        public void addScore(long index, double score) {
            reallocateGuard();
            scores.get(scores.size() - 1)[position] = score;
            indexes.get(scores.size() - 1)[position] = index;
            position += 1;
        }
    }

    ScoreStat scoreVsIter = new ScoreStat();

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
            scoreVsIter.reallocateGuard();
            scoreVsIter.addScore(iteration, score);
        }
    }

    public ScoreStat getScoreVsIter() {
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
        int largeBuckets = scoreVsIter.indexes.size();
        for (int j = 0; j < largeBuckets; ++j) {
            long[] indexes = scoreVsIter.indexes.get(j);
            double[] scores = scoreVsIter.scores.get(j);

            int effectiveLength = (j < largeBuckets -1) ? indexes.length : scoreVsIter.position;

            for (int i = 0; i < effectiveLength; ++i) {
                sb.append("\n").append(indexes[i]).append(delimiter).append(scores[i]);
            }
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
