/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.nlp.CbowInference;
import org.nd4j.linalg.api.ops.impl.nlp.CbowRound;
import org.nd4j.linalg.api.ops.impl.nlp.SkipGramInference;
import org.nd4j.linalg.api.ops.impl.nlp.SkipGramRound;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The word2vec rounds of the skipgram and cbow ops give what word2vec's training loops give, on CPU and CUDA alike:
 * <ul>
 *   <li>a hierarchic softmax step moves the vector that learns by the error it collected over all the points of the
 *   path, and syn1 by that point's own error (word2vec.c: neu1e += g * syn1, syn1 += g * syn0);</li>
 *   <li>negative sampling does the same against syn1Neg, once for the positive word and once for each negative;</li>
 *   <li>the codes may be of any integer type, and a code of -1 pads a row of codes;</li>
 *   <li>an inference vector is the only thing that learns: the tables are left as they are;</li>
 *   <li>iterations repeat the round with a fresh error vector each, and the learning rate decays after each one:
 *   alpha = (alpha - minLearningRate) / (iterations - iteration) + minLearningRate.</li>
 * </ul>
 * The ops take the optional inputs they do not use as empty arrays; every test runs such a configuration.
 * The negative table of the tests holds one word only, so that the words the rounds sample do not depend on the
 * random generator of the backend.
 */
@NativeTag
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class Word2VecRoundTest extends BaseNd4jTestWithBackends {

    private static final int VOCAB = 12;
    private static final int DIM = 24;
    private static final int EXP_LENGTH = 1000;
    private static final int MAX_EXP = 6;
    private static final float TOLERANCE = 1e-5f;
    private static final double ALPHA = 0.05;
    /** What skipgram and skipgram_inference decay towards when they are not given a minimum. */
    private static final double SKIPGRAM_MIN_ALPHA = 1e-4;
    /** What cbow_inference decays towards when it is not given a minimum. */
    private static final double CBOW_INFERENCE_MIN_ALPHA = 1e-3;
    /** The one word of the negative table, which the words that are trained for never are. */
    private static final int NEGATIVE_WORD = 11;
    private static final int NEGATIVE_TABLE_LENGTH = 64;

    // ------------------------------------------------------------------------------------------------------------
    // the model: the tables the ops train, and a Java copy of them that the references train
    // ------------------------------------------------------------------------------------------------------------

    private static final class Model {
        final float[][] syn0;
        final float[][] syn1;
        final float[][] syn1Neg;
        final float[] exp;

        Model(long seed) {
            syn0 = randomTable(seed);
            syn1 = randomTable(seed + 1);
            syn1Neg = randomTable(seed + 2);
            exp = expTable();
        }

        Model(Model other) {
            syn0 = copy(other.syn0);
            syn1 = copy(other.syn1);
            syn1Neg = copy(other.syn1Neg);
            exp = other.exp.clone();
        }

        INDArray syn0Array() {
            return Nd4j.createFromArray(copy(syn0));
        }

        INDArray syn1Array() {
            return Nd4j.createFromArray(copy(syn1));
        }

        INDArray syn1NegArray() {
            return Nd4j.createFromArray(copy(syn1Neg));
        }

        INDArray expArray() {
            return Nd4j.createFromArray(exp.clone());
        }
    }

    private static float[][] randomTable(long seed) {
        Random random = new Random(seed);
        float[][] table = new float[VOCAB][DIM];
        for (int r = 0; r < VOCAB; r++) {
            for (int c = 0; c < DIM; c++) {
                table[r][c] = random.nextFloat() - 0.5f;
            }
        }
        return table;
    }

    private static float[] randomVector(long seed) {
        Random random = new Random(seed);
        float[] vector = new float[DIM];
        for (int c = 0; c < DIM; c++) {
            vector[c] = (random.nextFloat() - 0.5f) / DIM;
        }
        return vector;
    }

    /** word2vec's table of sigmoid values over [-MAX_EXP, MAX_EXP). */
    private static float[] expTable() {
        float[] table = new float[EXP_LENGTH];
        for (int i = 0; i < EXP_LENGTH; i++) {
            double e = Math.exp((i / (double) EXP_LENGTH * 2 - 1) * MAX_EXP);
            table[i] = (float) (e / (e + 1));
        }
        return table;
    }

    private static float[][] copy(float[][] table) {
        float[][] copy = new float[table.length][];
        for (int r = 0; r < table.length; r++) {
            copy[r] = table[r].clone();
        }
        return copy;
    }

    private static INDArray negativeTable() {
        float[] table = new float[NEGATIVE_TABLE_LENGTH];
        Arrays.fill(table, NEGATIVE_WORD);
        return Nd4j.createFromArray(table);
    }

    // ------------------------------------------------------------------------------------------------------------
    // the references
    // ------------------------------------------------------------------------------------------------------------

    private static double dot(float[] a, float[] b) {
        double sum = 0;
        for (int e = 0; e < a.length; e++) {
            sum += (double) a[e] * b[e];
        }
        return sum;
    }

    /** One point of the path: the error it adds goes to neu1e, and syn1's row learns unless the vector is inferred. */
    private static void hierarchicStep(float[] exp, float[] vector, float[] syn1Row, int code, double alpha,
                                       double[] neu1e, boolean inference) {
        double dot = dot(vector, syn1Row);
        if (dot < -MAX_EXP || dot >= MAX_EXP) {
            return;
        }
        int idx = (int) (((float) dot + 6.0f) * ((float) exp.length / 6.0f / 2.0f));
        if (idx < 0 || idx >= exp.length) {
            return;
        }
        double g = (1 - code - exp[idx]) * alpha;
        for (int e = 0; e < vector.length; e++) {
            neu1e[e] += g * syn1Row[e];
            if (!inference) {
                syn1Row[e] += (float) (g * vector[e]);
            }
        }
    }

    /** One word of negative sampling: label 1 for the word that is trained for, 0 for a negative. */
    private static void negativeStep(float[] exp, float[] vector, float[] syn1NegRow, int label, double alpha,
                                     double[] neu1e, boolean inference) {
        double dot = dot(vector, syn1NegRow);
        double g;
        if (dot > MAX_EXP) {
            g = (label - 1) * alpha;
        } else if (dot < -MAX_EXP) {
            g = label * alpha;
        } else {
            int idx = (int) (((float) dot + 6.0f) * (((float) exp.length / 6.0f) / 2.0));
            if (idx < 0 || idx >= exp.length) {
                return;
            }
            g = (label - exp[idx]) * alpha;
        }
        for (int e = 0; e < vector.length; e++) {
            neu1e[e] += g * syn1NegRow[e];
            if (!inference) {
                syn1NegRow[e] += (float) (g * vector[e]);
            }
        }
    }

    /**
     * The rounds against one vector: hierarchic softmax over the points whose code is not a padding, then negative
     * sampling - the word that is trained for as the positive, the one word of the negative table as each negative.
     */
    private static void rounds(Model m, float[] vector, int[] points, int[] codes, int ngStarter, int nsRounds,
                               double alpha, double[] neu1e, boolean inference) {
        for (int h = 0; h < points.length; h++) {
            if (points[h] < 0 || codes[h] < 0) {
                continue;
            }
            hierarchicStep(m.exp, vector, m.syn1[points[h]], codes[h], alpha, neu1e, inference);
        }
        if (nsRounds > 0) {
            for (int r = 0; r < nsRounds + 1; r++) {
                int row = ngStarter;
                int label = 1;
                if (r > 0) {
                    row = NEGATIVE_WORD;
                    label = 0;
                    if (row == ngStarter) {
                        continue;
                    }
                }
                negativeStep(m.exp, vector, m.syn1Neg[row], label, alpha, neu1e, inference);
            }
        }
    }

    /** The table of negative words that the random generator of the ops picks from: every word of the vocabulary. */
    private static float[] spreadNegativeTable() {
        float[] table = new float[NEGATIVE_TABLE_LENGTH];
        for (int i = 0; i < table.length; i++) {
            table[i] = (i * 7 + 2) % VOCAB;
        }
        return table;
    }

    /**
     * The negative sampling of a round that follows the random generator of the ops: the word that is trained for as the
     * positive, then one negative drawn from the table for each of the rounds - a draw of the word that is trained for
     * is skipped. The random value goes on from one draw to the next, and from one round to the next.
     */
    private static void drawnNegativeRounds(Model m, float[] vector, int ngStarter, int nsRounds, double alpha,
                                            double[] neu1e, float[] table, long[] random, boolean inference) {
        for (int r = 0; r < nsRounds + 1; r++) {
            int row = ngStarter;
            int label = 1;
            if (r > 0) {
                random[0] = random[0] * 25214903917L + 11;
                row = (int) table[(int) Math.abs((random[0] >> 16) % table.length)];
                label = 0;
                if (row == ngStarter) {
                    continue;
                }
            }
            negativeStep(m.exp, vector, m.syn1Neg[row], label, alpha, neu1e, inference);
        }
    }

    /** A batch of skipgram rounds that infer, with negative sampling alone and the random values of the targets going on. */
    private static void skipgramBatchInferenceOfDrawnNegatives(Model m, float[] inference, int[] ngStarters,
                                                               int nsRounds, double[] alphas, int iterations,
                                                               double minAlpha, float[] table, long[] randoms) {
        double[] rates = alphas.clone();
        for (int iteration = 0; iteration < iterations; iteration++) {
            double[] sum = new double[DIM];
            for (int t = 0; t < ngStarters.length; t++) {
                double[] neu1e = new double[DIM];
                long[] random = {randoms[t]};
                drawnNegativeRounds(m, inference, ngStarters[t], nsRounds, rates[t], neu1e, table, random, true);
                randoms[t] = random[0];
                for (int e = 0; e < DIM; e++) {
                    sum[e] += neu1e[e];
                }
            }
            for (int e = 0; e < DIM; e++) {
                inference[e] += (float) sum[e];
            }
            for (int t = 0; t < rates.length; t++) {
                rates[t] = decay(rates[t], minAlpha, iterations, iteration);
            }
        }
    }

    private static double decay(double alpha, double minAlpha, int iterations, int iteration) {
        return (alpha - minAlpha) / (iterations - iteration) + minAlpha;
    }

    /** A skipgram round: the row of the target in syn0, or the inference vector, learns. */
    private static void skipgramRound(Model m, int target, float[] inference, int[] points, int[] codes, int ngStarter,
                                      int nsRounds, double alpha, int iterations, double minAlpha) {
        float[] vector = inference != null ? inference : m.syn0[target];
        for (int iteration = 0; iteration < iterations; iteration++) {
            double[] neu1e = new double[DIM];
            rounds(m, vector, points, codes, ngStarter, nsRounds, alpha, neu1e, inference != null);
            for (int e = 0; e < DIM; e++) {
                vector[e] += (float) neu1e[e];
            }
            alpha = decay(alpha, minAlpha, iterations, iteration);
        }
    }

    /** A batch of skipgram rounds that train: the targets one after the other, each with its own learning rate. */
    private static void skipgramBatchTraining(Model m, int[] targets, int[][] points, int[][] codes, int[] ngStarters,
                                              int nsRounds, double[] alphas) {
        for (int t = 0; t < targets.length; t++) {
            skipgramRound(m, targets[t], null, points[t], codes[t], ngStarters[t], nsRounds, alphas[t], 1,
                    SKIPGRAM_MIN_ALPHA);
        }
    }

    /**
     * A batch of skipgram rounds that infer: every target works against the same vector, the errors add up once per
     * iteration, and the learning rate of every target decays after each one.
     */
    private static void skipgramBatchInference(Model m, float[] inference, int[][] points, int[][] codes,
                                               int[] ngStarters, int nsRounds, double[] alphas, int iterations,
                                               double minAlpha) {
        double[] rates = alphas.clone();
        for (int iteration = 0; iteration < iterations; iteration++) {
            double[] sum = new double[DIM];
            for (int t = 0; t < points.length; t++) {
                double[] neu1e = new double[DIM];
                rounds(m, inference, points[t], codes[t], ngStarters[t], nsRounds, rates[t], neu1e, true);
                for (int e = 0; e < DIM; e++) {
                    sum[e] += neu1e[e];
                }
            }
            for (int e = 0; e < DIM; e++) {
                inference[e] += (float) sum[e];
            }
            for (int t = 0; t < rates.length; t++) {
                rates[t] = decay(rates[t], minAlpha, iterations, iteration);
            }
        }
    }

    /**
     * A cbow round on one window: the average of the words of the window (and of the inference vector, which is one more
     * member of it) is what the hierarchic softmax and the negative sampling see. The error moves the words of the
     * window that are not locked - from the first one, or from the labels at its end when the words are not trained -
     * or the inference vector, which is then all that learns.
     */
    private static void cbowRound(Model m, int[] context, int[] locked, int[] points, int[] codes, int ngStarter,
                                  int nsRounds, double alpha, int iterations, double minAlpha, boolean trainWords,
                                  int numLabels, float[] inference) {
        for (int iteration = 0; iteration < iterations; iteration++) {
            float[] neu1 = new float[DIM];
            int actual = 0;
            for (int word : context) {
                if (word < 0) {
                    continue;
                }
                for (int e = 0; e < DIM; e++) {
                    neu1[e] += m.syn0[word][e];
                }
                actual++;
            }
            if (inference != null) {
                for (int e = 0; e < DIM; e++) {
                    neu1[e] += inference[e];
                }
                actual++;
            }
            if (actual > 1) {
                for (int e = 0; e < DIM; e++) {
                    neu1[e] /= actual;
                }
            }

            double[] neu1e = new double[DIM];
            rounds(m, neu1, points, codes, ngStarter, nsRounds, alpha, neu1e, inference != null);

            if (inference == null) {
                int starter = trainWords ? 0 : Math.max(0, context.length - numLabels);
                for (int c = starter; c < context.length; c++) {
                    if (c < locked.length && locked[c] == 1) {
                        continue;
                    }
                    if (context[c] < 0) {
                        continue;
                    }
                    for (int e = 0; e < DIM; e++) {
                        m.syn0[context[c]][e] += (float) neu1e[e];
                    }
                }
            } else {
                for (int e = 0; e < DIM; e++) {
                    inference[e] += (float) neu1e[e];
                }
            }
            alpha = decay(alpha, minAlpha, iterations, iteration);
        }
    }

    /** A batch of cbow rounds on windows (padded with -1 words and codes) that train: one window after the other. */
    private static void cbowBatchTraining(Model m, int[][] contexts, int[][] locked, int[][] points, int[][] codes,
                                          int[] ngStarters, int nsRounds, double[] alphas) {
        for (int w = 0; w < contexts.length; w++) {
            cbowRound(m, contexts[w], locked[w], points[w], codes[w], ngStarters[w], nsRounds, alphas[w], 1,
                    CBOW_INFERENCE_MIN_ALPHA, true, 0, null);
        }
    }

    /** A batch of cbow rounds that infer: the windows in order, each moving the vector, then the rates decay. */
    private static void cbowBatchInference(Model m, float[] inference, int[][] contexts, int[][] locked,
                                           int[][] points, int[][] codes, int[] ngStarters, int nsRounds,
                                           double[] alphas, int iterations, double minAlpha) {
        double[] rates = alphas.clone();
        for (int iteration = 0; iteration < iterations; iteration++) {
            for (int w = 0; w < contexts.length; w++) {
                cbowRound(m, contexts[w], locked[w], points[w], codes[w], ngStarters[w], nsRounds, rates[w], 1,
                        minAlpha, true, 0, inference);
            }
            for (int w = 0; w < rates.length; w++) {
                rates[w] = decay(rates[w], minAlpha, iterations, iteration);
            }
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // running the ops
    // ------------------------------------------------------------------------------------------------------------

    private static void skipgram(INDArray target, INDArray ngStarter, INDArray indices, INDArray codes, INDArray syn0,
                                 INDArray syn1, INDArray syn1Neg, INDArray expTable, INDArray negTable, int nsRounds,
                                 INDArray alpha, INDArray inferenceVector, int iterations) {
        SkipGramRound op = SkipGramRound.builder()
                .target(target)
                .ngStarter(ngStarter)
                .indices(indices)
                .codes(codes)
                .syn0(syn0)
                .syn1(syn1)
                .syn1Neg(syn1Neg)
                .expTable(expTable)
                .negTable(negTable)
                .nsRounds(nsRounds)
                .alpha(alpha)
                .randomValue(randomValues(alpha))
                .inferenceVector(inferenceVector)
                .preciseMode(false)
                .numWorkers(1)
                .iterations(iterations)
                .build();
        Nd4j.getExecutioner().exec(op);
    }

    /** One random value for a scalar alpha, one for every target of an alpha vector. */
    private static INDArray randomValues(INDArray alpha) {
        if (alpha.isScalar()) {
            return Nd4j.scalar(119L);
        }
        long[] values = new long[(int) alpha.length()];
        for (int i = 0; i < values.length; i++) {
            values[i] = 119L + i;
        }
        return Nd4j.createFromArray(values);
    }

    private static void cbow(INDArray target, INDArray context, INDArray lockedWords, INDArray ngStarter,
                             INDArray indices, INDArray codes, INDArray syn0, INDArray syn1, INDArray syn1Neg,
                             INDArray expTable, INDArray negTable, int nsRounds, INDArray alpha,
                             INDArray inferenceVector, INDArray numLabels, boolean trainWords, int iterations,
                             double minLearningRate) {
        CbowRound op = CbowRound.builder()
                .target(target)
                .context(context)
                .lockedWords(lockedWords)
                .ngStarter(ngStarter)
                .syn0(syn0)
                .syn1(syn1)
                .syn1Neg(syn1Neg)
                .expTable(expTable)
                .negTable(negTable)
                .indices(indices)
                .codes(codes)
                .nsRounds(nsRounds)
                .alpha(alpha)
                .nextRandom(randomValues(alpha))
                .inferenceVector(inferenceVector)
                .numLabels(numLabels)
                .trainWords(trainWords)
                .numWorkers(1)
                .iterations(iterations)
                .minLearningRate(minLearningRate)
                .build();
        Nd4j.getExecutioner().exec(op);
    }

    private static int[][] padded(int[][] rows, int padding) {
        int width = 0;
        for (int[] row : rows) {
            width = Math.max(width, row.length);
        }
        int[][] result = new int[rows.length][width];
        for (int r = 0; r < rows.length; r++) {
            for (int c = 0; c < width; c++) {
                result[r][c] = c < rows[r].length ? rows[r][c] : padding;
            }
        }
        return result;
    }

    private static byte[] bytes(int[] values) {
        byte[] result = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = (byte) values[i];
        }
        return result;
    }

    // ------------------------------------------------------------------------------------------------------------
    // asserting
    // ------------------------------------------------------------------------------------------------------------

    private static void assertTable(String what, float[][] expected, INDArray actual) {
        float[] flat = actual.dup('c').data().asFloat();
        assertEquals((long) expected.length * expected[0].length, flat.length, what + ": length");
        for (int r = 0; r < expected.length; r++) {
            for (int c = 0; c < expected[r].length; c++) {
                assertEquals(expected[r][c], flat[r * expected[r].length + c], TOLERANCE,
                        what + " at [" + r + ", " + c + "]");
            }
        }
    }

    private static void assertVector(String what, float[] expected, INDArray actual) {
        float[] flat = actual.dup('c').data().asFloat();
        assertEquals(expected.length, flat.length, what + ": length");
        for (int c = 0; c < expected.length; c++) {
            assertEquals(expected[c], flat[c], TOLERANCE, what + " at " + c);
        }
    }

    private static void assertModel(String what, Model expected, INDArray syn0, INDArray syn1, INDArray syn1Neg) {
        assertTable(what + ": syn0", expected.syn0, syn0);
        if (syn1 != null) {
            assertTable(what + ": syn1", expected.syn1, syn1);
        }
        if (syn1Neg != null) {
            assertTable(what + ": syn1Neg", expected.syn1Neg, syn1Neg);
        }
    }

    // ------------------------------------------------------------------------------------------------------------
    // skipgram
    // ------------------------------------------------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void skipgramRoundReadsCodesOfEveryIntegerType(Nd4jBackend backend) {
        int target = 3;
        int[] points = {1, 5, 7};
        int[] codes = {0, 1, 1};
        INDArray[] codeArrays = {Nd4j.createFromArray(bytes(codes)), Nd4j.createFromArray(codes),
                Nd4j.createFromArray(new long[]{0, 1, 1})};
        for (INDArray codeArray : codeArrays) {
            Model model = new Model(11);
            Model expected = new Model(model);
            skipgramRound(expected, target, null, points, codes, -1, 0, ALPHA, 1, SKIPGRAM_MIN_ALPHA);

            INDArray syn0 = model.syn0Array();
            INDArray syn1 = model.syn1Array();
            skipgram(Nd4j.scalar(target), Nd4j.empty(DataType.INT32), Nd4j.createFromArray(points), codeArray, syn0,
                    syn1, Nd4j.empty(DataType.FLOAT), model.expArray(), Nd4j.empty(DataType.FLOAT), 0,
                    Nd4j.scalar(ALPHA), Nd4j.empty(DataType.FLOAT), 1);
            assertModel("skipgram with " + codeArray.dataType() + " codes", expected, syn0, syn1, null);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void skipgramRoundOfSeveralIterations(Nd4jBackend backend) {
        // every iteration starts from a zero error vector and the rate decays after it: 0.05, 0.0167, 0.0042
        int target = 6;
        int[] points = {2, 4, 9};
        int[] codes = {1, 0, 1};
        Model model = new Model(12);
        Model expected = new Model(model);
        skipgramRound(expected, target, null, points, codes, -1, 0, ALPHA, 3, SKIPGRAM_MIN_ALPHA);

        INDArray syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        skipgram(Nd4j.scalar(target), Nd4j.empty(DataType.INT32), Nd4j.createFromArray(points),
                Nd4j.createFromArray(bytes(codes)), syn0, syn1, Nd4j.empty(DataType.FLOAT), model.expArray(),
                Nd4j.empty(DataType.FLOAT), 0, Nd4j.scalar(ALPHA), Nd4j.empty(DataType.FLOAT), 3);
        assertModel("skipgram of 3 iterations", expected, syn0, syn1, null);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void skipgramRoundOfNegativeSampling(Nd4jBackend backend) {
        // negative sampling alone: no syn1, no points, no codes
        int target = 2;
        int ngStarter = 4;
        Model model = new Model(13);
        Model expected = new Model(model);
        skipgramRound(expected, target, null, new int[0], new int[0], ngStarter, 2, ALPHA, 2, SKIPGRAM_MIN_ALPHA);

        INDArray syn0 = model.syn0Array();
        INDArray syn1Neg = model.syn1NegArray();
        skipgram(Nd4j.scalar(target), Nd4j.scalar(ngStarter), Nd4j.empty(DataType.INT32),
                Nd4j.empty(DataType.INT8), syn0, Nd4j.empty(DataType.FLOAT), syn1Neg, model.expArray(),
                negativeTable(), 2, Nd4j.scalar(ALPHA), Nd4j.empty(DataType.FLOAT), 2);
        assertModel("skipgram of negative sampling", expected, syn0, null, syn1Neg);

        // hierarchic softmax and negative sampling together
        int[] points = {2, 6};
        int[] codes = {1, 0};
        model = new Model(14);
        expected = new Model(model);
        skipgramRound(expected, target, null, points, codes, ngStarter, 3, ALPHA, 1, SKIPGRAM_MIN_ALPHA);

        syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        syn1Neg = model.syn1NegArray();
        skipgram(Nd4j.scalar(target), Nd4j.scalar(ngStarter), Nd4j.createFromArray(points),
                Nd4j.createFromArray(codes), syn0, syn1, syn1Neg, model.expArray(), negativeTable(), 3,
                Nd4j.scalar(ALPHA), Nd4j.empty(DataType.FLOAT), 1);
        assertModel("skipgram of both", expected, syn0, syn1, syn1Neg);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void negativeSamplingWithoutItsTableIsRejected(Nd4jBackend backend) {
        // an empty array is an absent one: asking for negative samples without a table is an error, not a modulo by zero
        Model model = new Model(32);
        INDArray syn0 = model.syn0Array();
        INDArray syn1Neg = model.syn1NegArray();
        assertThrows(RuntimeException.class, () -> skipgram(Nd4j.scalar(2), Nd4j.scalar(4),
                Nd4j.empty(DataType.INT32), Nd4j.empty(DataType.INT8), syn0, Nd4j.empty(DataType.FLOAT), syn1Neg,
                model.expArray(), Nd4j.empty(DataType.FLOAT), 2, Nd4j.scalar(ALPHA), Nd4j.empty(DataType.FLOAT), 1));
        assertTable("syn0 after the rejected round", model.syn0, syn0);
        assertTable("syn1Neg after the rejected round", model.syn1Neg, syn1Neg);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void roundsOfDoubleTables(Nd4jBackend backend) {
        // the tables, the exp table, the negative table and the empty arrays are all DOUBLE
        int[] points = {2, 6};
        int[] codes = {1, 0};
        int ngStarter = 4;

        Model model = new Model(30);
        Model expected = new Model(model);
        skipgramRound(expected, 2, null, points, codes, ngStarter, 2, ALPHA, 2, SKIPGRAM_MIN_ALPHA);
        INDArray syn0 = model.syn0Array().castTo(DataType.DOUBLE);
        INDArray syn1 = model.syn1Array().castTo(DataType.DOUBLE);
        INDArray syn1Neg = model.syn1NegArray().castTo(DataType.DOUBLE);
        skipgram(Nd4j.scalar(2), Nd4j.scalar(ngStarter), Nd4j.createFromArray(points),
                Nd4j.createFromArray(bytes(codes)), syn0, syn1, syn1Neg, model.expArray().castTo(DataType.DOUBLE),
                negativeTable().castTo(DataType.DOUBLE), 2, Nd4j.scalar(ALPHA), Nd4j.empty(DataType.DOUBLE), 2);
        assertModel("skipgram of DOUBLE tables", expected, syn0, syn1, syn1Neg);

        int[] context = {3, 6, 9};
        int[] locked = {0, 0, 0};
        model = new Model(31);
        expected = new Model(model);
        cbowRound(expected, context, locked, points, codes, ngStarter, 2, ALPHA, 1, CBOW_INFERENCE_MIN_ALPHA, true, 0,
                null);
        syn0 = model.syn0Array().castTo(DataType.DOUBLE);
        syn1 = model.syn1Array().castTo(DataType.DOUBLE);
        syn1Neg = model.syn1NegArray().castTo(DataType.DOUBLE);
        cbow(Nd4j.scalar(0), Nd4j.createFromArray(context), Nd4j.createFromArray(locked), Nd4j.scalar(ngStarter),
                Nd4j.createFromArray(points), Nd4j.createFromArray(bytes(codes)), syn0, syn1, syn1Neg,
                model.expArray().castTo(DataType.DOUBLE), negativeTable().castTo(DataType.DOUBLE), 2,
                Nd4j.scalar(ALPHA), Nd4j.empty(DataType.DOUBLE), Nd4j.empty(DataType.INT32), true, 1,
                CBOW_INFERENCE_MIN_ALPHA);
        assertModel("cbow of DOUBLE tables", expected, syn0, syn1, syn1Neg);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void skipgramRoundOfAnInferenceVectorLeavesTheTables(Nd4jBackend backend) {
        int[] points = {1, 3, 8};
        int[] codes = {0, 0, 1};
        int ngStarter = 5;
        Model model = new Model(15);
        Model expected = new Model(model);
        float[] vector = randomVector(77);
        float[] expectedVector = vector.clone();
        skipgramRound(expected, -1, expectedVector, points, codes, ngStarter, 2, ALPHA, 3, SKIPGRAM_MIN_ALPHA);

        INDArray syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        INDArray syn1Neg = model.syn1NegArray();
        INDArray inference = Nd4j.createFromArray(vector.clone());
        // the target is not read when there is an inference vector, but a scalar tells the op this is one round
        skipgram(Nd4j.scalar(0), Nd4j.scalar(ngStarter), Nd4j.createFromArray(points),
                Nd4j.createFromArray(bytes(codes)), syn0, syn1, syn1Neg, model.expArray(), negativeTable(), 2,
                Nd4j.scalar(ALPHA), inference, 3);

        assertVector("the inference vector", expectedVector, inference);
        assertModel("the tables of an inference", expected, syn0, syn1, syn1Neg);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void skipgramBatchOfTargetsWithCodesOfDifferentLengths(Nd4jBackend backend) {
        // the codes of a batch are padded to the longest with -1; the padding must not train anything. One target twice.
        int[] targets = {2, 9, 4, 4, 0};
        int[][] points = {{1, 3, 6}, {2, 4}, {5}, {7, 8, 10}, {1, 2}};
        int[][] codes = {{0, 1, 0}, {1, 1}, {0}, {1, 0, 1}, {0, 0}};
        double[] alphas = {0.05, 0.04, 0.03, 0.02, 0.01};
        int[] ngStarters = new int[targets.length];

        Model model = new Model(16);
        Model expected = new Model(model);
        skipgramBatchTraining(expected, targets, points, codes, ngStarters, 0, alphas);

        INDArray syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        skipgram(Nd4j.createFromArray(targets), Nd4j.empty(DataType.INT32),
                Nd4j.createFromArray(padded(points, 0)), Nd4j.createFromArray(padded(codes, -1)), syn0, syn1,
                Nd4j.empty(DataType.FLOAT), model.expArray(), Nd4j.empty(DataType.FLOAT), 0,
                Nd4j.createFromArray(alphas), Nd4j.empty(DataType.FLOAT), 1);
        assertModel("skipgram batch", expected, syn0, syn1, null);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void skipgramBatchOfNegativeSampling(Nd4jBackend backend) {
        int[] targets = {1, 7, 3, 10};
        int[] ngStarters = {4, 5, 0, 8};
        double[] alphas = {0.05, 0.04, 0.03, 0.02};

        Model model = new Model(17);
        Model expected = new Model(model);
        int[][] none = new int[targets.length][0];
        skipgramBatchTraining(expected, targets, none, none, ngStarters, 2, alphas);

        INDArray syn0 = model.syn0Array();
        INDArray syn1Neg = model.syn1NegArray();
        skipgram(Nd4j.createFromArray(targets), Nd4j.createFromArray(ngStarters), Nd4j.empty(DataType.INT32),
                Nd4j.empty(DataType.INT8), syn0, Nd4j.empty(DataType.FLOAT), syn1Neg, model.expArray(),
                negativeTable(), 2, Nd4j.createFromArray(alphas), Nd4j.empty(DataType.FLOAT), 1);
        assertModel("skipgram batch of negative sampling", expected, syn0, null, syn1Neg);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void skipgramBatchOfAnInferenceVector(Nd4jBackend backend) {
        int[] targets = {2, 9, 4};
        int[][] points = {{1, 3, 6}, {2, 4}, {5, 8}};
        int[][] codes = {{0, 1, 0}, {1, 1}, {0, 1}};
        int[] ngStarters = {3, 6, 9};
        double[] alphas = {0.05, 0.04, 0.03};

        Model model = new Model(18);
        Model expected = new Model(model);
        float[] vector = randomVector(78);
        float[] expectedVector = vector.clone();
        skipgramBatchInference(expected, expectedVector, points, codes, ngStarters, 1, alphas, 2,
                SKIPGRAM_MIN_ALPHA);

        INDArray syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        INDArray syn1Neg = model.syn1NegArray();
        INDArray inference = Nd4j.createFromArray(vector.clone());
        skipgram(Nd4j.createFromArray(targets), Nd4j.createFromArray(ngStarters),
                Nd4j.createFromArray(padded(points, 0)), Nd4j.createFromArray(padded(codes, -1)), syn0, syn1, syn1Neg,
                model.expArray(), negativeTable(), 1, Nd4j.createFromArray(alphas), inference, 2);

        assertVector("the inference vector of a batch", expectedVector, inference);
        assertModel("the tables of a batch inference", expected, syn0, syn1, syn1Neg);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void skipgramInferenceOpTrainsAndInfers(Nd4jBackend backend) {
        int target = 5;
        int[] points = {2, 7};
        int[] codes = {1, 0};

        // training: the row of the target moves, with codes and indices that travel as integer arguments
        Model model = new Model(19);
        Model expected = new Model(model);
        skipgramRound(expected, target, null, points, codes, -1, 0, ALPHA, 1, SKIPGRAM_MIN_ALPHA);
        INDArray syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        Nd4j.getExecutioner().exec(SkipGramInference.builder()
                .target(target)
                .iteration(1)
                .ngStarter(-1)
                .syn0(syn0)
                .syn1(syn1)
                .syn1Neg(Nd4j.empty(DataType.FLOAT))
                .expTable(model.expArray())
                .negTable(Nd4j.empty(DataType.FLOAT))
                .nsRounds(0)
                .indices(points)
                .codes(bytes(codes))
                .alpha(new double[]{ALPHA})
                .randomValue(119)
                .inferenceVector(Nd4j.empty(DataType.FLOAT))
                .preciseMode(false)
                .numWorkers(1)
                .build());
        assertModel("skipgram_inference that trains", expected, syn0, syn1, null);

        // inference over 3 iterations: only the vector moves
        model = new Model(20);
        expected = new Model(model);
        float[] vector = randomVector(79);
        float[] expectedVector = vector.clone();
        skipgramRound(expected, -1, expectedVector, points, codes, -1, 0, ALPHA, 3, SKIPGRAM_MIN_ALPHA);
        syn0 = model.syn0Array();
        syn1 = model.syn1Array();
        INDArray inference = Nd4j.createFromArray(vector.clone());
        Nd4j.getExecutioner().exec(SkipGramInference.builder()
                .target(-1)
                .iteration(3)
                .ngStarter(-1)
                .syn0(syn0)
                .syn1(syn1)
                .syn1Neg(Nd4j.empty(DataType.FLOAT))
                .expTable(model.expArray())
                .negTable(Nd4j.empty(DataType.FLOAT))
                .nsRounds(0)
                .indices(points)
                .codes(bytes(codes))
                .alpha(new double[]{ALPHA})
                .randomValue(119)
                .inferenceVector(inference)
                .preciseMode(false)
                .numWorkers(1)
                .build());
        assertVector("the inference vector of skipgram_inference", expectedVector, inference);
        assertModel("the tables of skipgram_inference", expected, syn0, syn1, null);
    }

    // ------------------------------------------------------------------------------------------------------------
    // cbow
    // ------------------------------------------------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cbowRoundOfAWindow(Nd4jBackend backend) {
        // the window is averaged, the word 8 is locked and does not learn; codes of two integer types
        int[] context = {2, 5, 8};
        int[] locked = {0, 0, 1};
        int[] points = {1, 4};
        int[] codes = {0, 1};
        INDArray[] codeArrays = {Nd4j.createFromArray(bytes(codes)), Nd4j.createFromArray(codes)};
        for (INDArray codeArray : codeArrays) {
            Model model = new Model(21);
            Model expected = new Model(model);
            cbowRound(expected, context, locked, points, codes, -1, 0, ALPHA, 1, CBOW_INFERENCE_MIN_ALPHA, true, 0,
                    null);

            INDArray syn0 = model.syn0Array();
            INDArray syn1 = model.syn1Array();
            cbow(Nd4j.scalar(0), Nd4j.createFromArray(context), Nd4j.createFromArray(locked),
                    Nd4j.empty(DataType.INT32), Nd4j.createFromArray(points), codeArray, syn0, syn1,
                    Nd4j.empty(DataType.FLOAT), model.expArray(), Nd4j.empty(DataType.FLOAT), 0, Nd4j.scalar(ALPHA),
                    Nd4j.empty(DataType.FLOAT), Nd4j.empty(DataType.INT32), true, 1, CBOW_INFERENCE_MIN_ALPHA);
            assertModel("cbow with " + codeArray.dataType() + " codes", expected, syn0, syn1, null);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cbowRoundOfHierarchicSoftmaxAndNegativeSampling(Nd4jBackend backend) {
        // the configuration of DL4J's CBOW with both: INT8 codes, the word that is trained for as the positive
        int[] context = {3, 6, 9};
        int[] locked = {0, 0, 0};
        int[] points = {2, 7};
        int[] codes = {1, 0};
        int ngStarter = 5;
        Model model = new Model(29);
        Model expected = new Model(model);
        cbowRound(expected, context, locked, points, codes, ngStarter, 2, ALPHA, 1, CBOW_INFERENCE_MIN_ALPHA, true, 0,
                null);

        INDArray syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        INDArray syn1Neg = model.syn1NegArray();
        cbow(Nd4j.scalar(0), Nd4j.createFromArray(context), Nd4j.createFromArray(locked), Nd4j.scalar(ngStarter),
                Nd4j.createFromArray(points), Nd4j.createFromArray(bytes(codes)), syn0, syn1, syn1Neg,
                model.expArray(), negativeTable(), 2, Nd4j.scalar(ALPHA), Nd4j.empty(DataType.FLOAT),
                Nd4j.empty(DataType.INT32), true, 1, CBOW_INFERENCE_MIN_ALPHA);
        assertModel("cbow of both", expected, syn0, syn1, syn1Neg);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cbowRoundThatTrainsTheLabelsOnly(Nd4jBackend backend) {
        // the last word of the window is a label: without trainWords it is the only one that learns
        int[] context = {1, 4, 9};
        int[] locked = {0, 0, 0};
        int[] points = {3, 6};
        int[] codes = {1, 1};
        Model model = new Model(22);
        Model expected = new Model(model);
        cbowRound(expected, context, locked, points, codes, -1, 0, ALPHA, 1, CBOW_INFERENCE_MIN_ALPHA, false, 1, null);

        INDArray syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        cbow(Nd4j.scalar(0), Nd4j.createFromArray(context), Nd4j.createFromArray(locked), Nd4j.empty(DataType.INT32),
                Nd4j.createFromArray(points), Nd4j.createFromArray(bytes(codes)), syn0, syn1,
                Nd4j.empty(DataType.FLOAT), model.expArray(), Nd4j.empty(DataType.FLOAT), 0, Nd4j.scalar(ALPHA),
                Nd4j.empty(DataType.FLOAT), Nd4j.scalar(1), false, 1, CBOW_INFERENCE_MIN_ALPHA);
        assertModel("cbow that trains the labels only", expected, syn0, syn1, null);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cbowRoundOfAnInferenceVector(Nd4jBackend backend) {
        // the vector is one more member of the window; over 3 iterations only it moves
        int[] context = {0, 3, 10};
        int[] locked = {0, 0, 0};
        int[] points = {2, 5, 7};
        int[] codes = {0, 1, 0};
        Model model = new Model(23);
        Model expected = new Model(model);
        float[] vector = randomVector(80);
        float[] expectedVector = vector.clone();
        cbowRound(expected, context, locked, points, codes, 6, 1, ALPHA, 3, 1e-4, true, 0, expectedVector);

        INDArray syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        INDArray syn1Neg = model.syn1NegArray();
        INDArray inference = Nd4j.createFromArray(vector.clone());
        cbow(Nd4j.scalar(0), Nd4j.createFromArray(context), Nd4j.createFromArray(locked), Nd4j.scalar(6),
                Nd4j.createFromArray(points), Nd4j.createFromArray(bytes(codes)), syn0, syn1, syn1Neg,
                model.expArray(), negativeTable(), 1, Nd4j.scalar(ALPHA), inference, Nd4j.empty(DataType.INT32), true,
                3, 1e-4);

        assertVector("the inference vector of cbow", expectedVector, inference);
        assertModel("the tables of cbow with an inference vector", expected, syn0, syn1, syn1Neg);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cbowBatchOfWindowsPaddedWithNegativeWords(Nd4jBackend backend) {
        int[][] contexts = {{1, 2, 3}, {6, 7}, {0, 3, 5, 8}, {11, 2}};
        int[][] locked = {{0, 0, 0}, {0, 1}, {0, 0, 0, 0}, {0, 0}};
        int[][] points = {{4, 5}, {2, 9, 10}, {1}, {4, 6}};
        int[][] codes = {{0, 1}, {1, 1, 0}, {1}, {0, 0}};
        int[] ngStarters = {1, 2, 3, 4};
        double[] alphas = {0.05, 0.04, 0.03, 0.02};

        Model model = new Model(24);
        Model expected = new Model(model);
        cbowBatchTraining(expected, contexts, locked, points, codes, ngStarters, 0, alphas);

        INDArray syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        cbow(Nd4j.createFromArray(ngStarters), Nd4j.createFromArray(padded(contexts, -1)),
                Nd4j.createFromArray(padded(locked, -1)), Nd4j.createFromArray(ngStarters),
                Nd4j.createFromArray(padded(points, 0)), Nd4j.createFromArray(padded(codes, -1)), syn0, syn1,
                Nd4j.empty(DataType.FLOAT), model.expArray(), Nd4j.empty(DataType.FLOAT), 0,
                Nd4j.createFromArray(alphas), Nd4j.empty(DataType.FLOAT), Nd4j.empty(DataType.INT32), true, 1,
                CBOW_INFERENCE_MIN_ALPHA);
        assertModel("cbow batch", expected, syn0, syn1, null);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cbowBatchOfNegativeSamplingAlone(Nd4jBackend backend) {
        // without hierarchic softmax the batch has no indices and no codes
        int[][] contexts = {{1, 2, 3}, {6, 7, 0}, {5, 3, 8}};
        int[][] locked = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        int[] ngStarters = {4, 5, 9};
        double[] alphas = {0.05, 0.04, 0.03};
        int[][] none = new int[contexts.length][0];

        Model model = new Model(25);
        Model expected = new Model(model);
        cbowBatchTraining(expected, contexts, locked, none, none, ngStarters, 2, alphas);

        INDArray syn0 = model.syn0Array();
        INDArray syn1Neg = model.syn1NegArray();
        cbow(Nd4j.createFromArray(ngStarters), Nd4j.createFromArray(padded(contexts, -1)),
                Nd4j.createFromArray(padded(locked, -1)), Nd4j.createFromArray(ngStarters),
                Nd4j.empty(DataType.INT32), Nd4j.empty(DataType.INT8), syn0, Nd4j.empty(DataType.FLOAT), syn1Neg,
                model.expArray(), negativeTable(), 2, Nd4j.createFromArray(alphas), Nd4j.empty(DataType.FLOAT),
                Nd4j.empty(DataType.INT32), true, 1, CBOW_INFERENCE_MIN_ALPHA);
        assertModel("cbow batch of negative sampling", expected, syn0, null, syn1Neg);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cbowBatchOfAnInferenceVector(Nd4jBackend backend) {
        int[][] contexts = {{1, 2, 3}, {6, 7}, {0, 3, 5}};
        int[][] locked = {{0, 0, 0}, {0, 0}, {0, 0, 0}};
        int[][] points = {{4, 5}, {2, 9}, {1, 3}};
        int[][] codes = {{0, 1}, {1, 0}, {1, 1}};
        int[] ngStarters = {1, 2, 3};
        double[] alphas = {0.05, 0.04, 0.03};

        Model model = new Model(26);
        Model expected = new Model(model);
        float[] vector = randomVector(81);
        float[] expectedVector = vector.clone();
        cbowBatchInference(expected, expectedVector, contexts, locked, points, codes, ngStarters, 0, alphas, 2, 1e-4);

        INDArray syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        INDArray inference = Nd4j.createFromArray(vector.clone());
        cbow(Nd4j.createFromArray(ngStarters), Nd4j.createFromArray(padded(contexts, -1)),
                Nd4j.createFromArray(padded(locked, -1)), Nd4j.createFromArray(ngStarters),
                Nd4j.createFromArray(padded(points, 0)), Nd4j.createFromArray(padded(codes, -1)), syn0, syn1,
                Nd4j.empty(DataType.FLOAT), model.expArray(), Nd4j.empty(DataType.FLOAT), 0,
                Nd4j.createFromArray(alphas), inference, Nd4j.empty(DataType.INT32), true, 2, 1e-4);

        assertVector("the inference vector of a cbow batch", expectedVector, inference);
        assertModel("the tables of a cbow batch inference", expected, syn0, syn1, null);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cbowInferenceOpTrainsAndInfers(Nd4jBackend backend) {
        int[] context = {2, 5, 9};
        int[] locked = {0, 1, 0};
        int[] points = {1, 6};
        int[] codes = {0, 1};

        // training: the words of the window that are not locked move
        Model model = new Model(27);
        Model expected = new Model(model);
        cbowRound(expected, context, locked, points, codes, -1, 0, ALPHA, 1, CBOW_INFERENCE_MIN_ALPHA, true, 0, null);
        INDArray syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        Nd4j.getExecutioner().exec(CbowInference.builder()
                .target(0)
                .ngStarter(-1)
                .syn0(syn0)
                .syn1(syn1)
                .syn1Neg(Nd4j.empty(DataType.FLOAT))
                .expTable(model.expArray())
                .negTable(Nd4j.empty(DataType.FLOAT))
                .nsRounds(0)
                .indices(points)
                .lockedWords(locked)
                .context(context)
                .codes(bytes(codes))
                .alpha(ALPHA)
                .randomValue(119)
                .inferenceVector(Nd4j.empty(DataType.FLOAT))
                .preciseMode(false)
                .trainWords(true)
                .numWorkers(1)
                .numLabels(0)
                .iterations(1)
                .build());
        assertModel("cbow_inference that trains", expected, syn0, syn1, null);

        // inference over 3 iterations: only the vector moves
        model = new Model(28);
        expected = new Model(model);
        float[] vector = randomVector(82);
        float[] expectedVector = vector.clone();
        cbowRound(expected, context, locked, points, codes, -1, 0, ALPHA, 3, CBOW_INFERENCE_MIN_ALPHA, true, 0,
                expectedVector);
        syn0 = model.syn0Array();
        syn1 = model.syn1Array();
        INDArray inference = Nd4j.createFromArray(vector.clone());
        Nd4j.getExecutioner().exec(CbowInference.builder()
                .target(0)
                .ngStarter(-1)
                .syn0(syn0)
                .syn1(syn1)
                .syn1Neg(Nd4j.empty(DataType.FLOAT))
                .expTable(model.expArray())
                .negTable(Nd4j.empty(DataType.FLOAT))
                .nsRounds(0)
                .indices(points)
                .lockedWords(locked)
                .context(context)
                .codes(bytes(codes))
                .alpha(ALPHA)
                .randomValue(119)
                .inferenceVector(inference)
                .preciseMode(false)
                .trainWords(true)
                .numWorkers(1)
                .numLabels(0)
                .iterations(3)
                .build());
        assertVector("the inference vector of cbow_inference", expectedVector, inference);
        assertModel("the tables of cbow_inference", expected, syn0, syn1, null);
    }

    // ------------------------------------------------------------------------------------------------------------
    // what the helpers do with the arguments they are given
    // ------------------------------------------------------------------------------------------------------------

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void skipgramInferenceOpOfNegativeSamplingAlone(Nd4jBackend backend) {
        // DL4J's skipgram without hierarchic softmax gives the op no codes and no indices: nothing to make an array of
        int target = 5;
        int ngStarter = 4;

        // training
        Model model = new Model(40);
        Model expected = new Model(model);
        skipgramRound(expected, target, null, new int[0], new int[0], ngStarter, 2, ALPHA, 1, SKIPGRAM_MIN_ALPHA);
        INDArray syn0 = model.syn0Array();
        INDArray syn1Neg = model.syn1NegArray();
        Nd4j.getExecutioner().exec(SkipGramInference.builder()
                .target(target)
                .iteration(1)
                .ngStarter(ngStarter)
                .syn0(syn0)
                .syn1(Nd4j.empty(DataType.FLOAT))
                .syn1Neg(syn1Neg)
                .expTable(model.expArray())
                .negTable(negativeTable())
                .nsRounds(2)
                .indices(new int[0])
                .codes(new byte[0])
                .alpha(new double[]{ALPHA})
                .randomValue(119)
                .inferenceVector(Nd4j.empty(DataType.FLOAT))
                .preciseMode(false)
                .numWorkers(1)
                .build());
        assertModel("skipgram_inference of negative sampling that trains", expected, syn0, null, syn1Neg);

        // inference over 2 iterations
        model = new Model(41);
        expected = new Model(model);
        float[] vector = randomVector(83);
        float[] expectedVector = vector.clone();
        skipgramRound(expected, -1, expectedVector, new int[0], new int[0], ngStarter, 2, ALPHA, 2,
                SKIPGRAM_MIN_ALPHA);
        syn0 = model.syn0Array();
        syn1Neg = model.syn1NegArray();
        INDArray inference = Nd4j.createFromArray(vector.clone());
        Nd4j.getExecutioner().exec(SkipGramInference.builder()
                .target(-1)
                .iteration(2)
                .ngStarter(ngStarter)
                .syn0(syn0)
                .syn1(Nd4j.empty(DataType.FLOAT))
                .syn1Neg(syn1Neg)
                .expTable(model.expArray())
                .negTable(negativeTable())
                .nsRounds(2)
                .indices(new int[0])
                .codes(new byte[0])
                .alpha(new double[]{ALPHA})
                .randomValue(119)
                .inferenceVector(inference)
                .preciseMode(false)
                .numWorkers(1)
                .build());
        assertVector("the inference vector of skipgram_inference of negative sampling", expectedVector, inference);
        assertModel("the tables of skipgram_inference of negative sampling", expected, syn0, null, syn1Neg);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void cbowRoundOfAWindowPaddedWithNegativeWords(Nd4jBackend backend) {
        // the values that pad a window are negative: they add nothing, and the average is over the words that are there
        int[] context = {3, -1, 6, -1, 9};
        int[] locked = {0, -1, 0, -1, 0};
        int[] points = {1, 4};
        int[] codes = {0, 1};
        Model model = new Model(42);
        Model expected = new Model(model);
        cbowRound(expected, context, locked, points, codes, -1, 0, ALPHA, 2, CBOW_INFERENCE_MIN_ALPHA, true, 0, null);

        INDArray syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        cbow(Nd4j.scalar(0), Nd4j.createFromArray(context), Nd4j.createFromArray(locked), Nd4j.empty(DataType.INT32),
                Nd4j.createFromArray(points), Nd4j.createFromArray(bytes(codes)), syn0, syn1,
                Nd4j.empty(DataType.FLOAT), model.expArray(), Nd4j.empty(DataType.FLOAT), 0, Nd4j.scalar(ALPHA),
                Nd4j.empty(DataType.FLOAT), Nd4j.empty(DataType.INT32), true, 2, CBOW_INFERENCE_MIN_ALPHA);
        assertModel("cbow of a padded window", expected, syn0, syn1, null);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void skipgramBatchInferenceGoesOnWithItsRandomValues(Nd4jBackend backend) {
        // The negatives are drawn from a table that holds every word; the random value of a target goes on from one
        // iteration to the next, so the negatives of the second iteration are not those of the first.
        int[] targets = {0, 0, 0};
        int[] ngStarters = {3, 6, 9};
        double[] alphas = {0.05, 0.04, 0.03};
        float[] table = spreadNegativeTable();
        long[] randoms = {119L, 120L, 121L};
        int nsRounds = 3;
        int iterations = 3;

        Model model = new Model(43);
        Model expected = new Model(model);
        float[] vector = randomVector(84);
        float[] expectedVector = vector.clone();
        skipgramBatchInferenceOfDrawnNegatives(expected, expectedVector, ngStarters, nsRounds, alphas, iterations,
                SKIPGRAM_MIN_ALPHA, table, randoms.clone());

        INDArray syn0 = model.syn0Array();
        INDArray syn1Neg = model.syn1NegArray();
        INDArray inference = Nd4j.createFromArray(vector.clone());
        skipgram(Nd4j.createFromArray(targets), Nd4j.createFromArray(ngStarters), Nd4j.empty(DataType.INT32),
                Nd4j.empty(DataType.INT8), syn0, Nd4j.empty(DataType.FLOAT), syn1Neg, model.expArray(),
                Nd4j.createFromArray(table), nsRounds, Nd4j.createFromArray(alphas), inference, iterations);

        assertVector("the inference vector of drawn negatives", expectedVector, inference);
        assertModel("the tables of drawn negatives", expected, syn0, null, syn1Neg);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void argumentsThatAreNotRowsOfTheTablesAreRejectedBeforeAnythingTrains(Nd4jBackend backend) {
        // the last target of the batch is not a row of syn0: the targets before it must not have trained
        Model model = new Model(44);
        INDArray syn0 = model.syn0Array();
        INDArray syn1 = model.syn1Array();
        assertThrows(RuntimeException.class, () -> skipgram(Nd4j.createFromArray(new int[]{2, 5, VOCAB}),
                Nd4j.empty(DataType.INT32), Nd4j.createFromArray(new int[][]{{1, 3}, {2, 4}, {5, 6}}),
                Nd4j.createFromArray(new int[][]{{0, 1}, {1, 1}, {0, 1}}), syn0, syn1, Nd4j.empty(DataType.FLOAT),
                model.expArray(), Nd4j.empty(DataType.FLOAT), 0, Nd4j.createFromArray(new double[]{0.05, 0.04, 0.03}),
                Nd4j.empty(DataType.FLOAT), 1));
        assertTable("syn0 after the rejected batch", model.syn0, syn0);
        assertTable("syn1 after the rejected batch", model.syn1, syn1);

        // the word to sample negatives for is not a row of syn1Neg
        INDArray syn1Neg = model.syn1NegArray();
        assertThrows(RuntimeException.class, () -> skipgram(Nd4j.scalar(2), Nd4j.scalar(VOCAB),
                Nd4j.empty(DataType.INT32), Nd4j.empty(DataType.INT8), syn0, Nd4j.empty(DataType.FLOAT), syn1Neg,
                model.expArray(), negativeTable(), 2, Nd4j.scalar(ALPHA), Nd4j.empty(DataType.FLOAT), 1));
        assertTable("syn0 after the rejected round", model.syn0, syn0);
        assertTable("syn1Neg after the rejected round", model.syn1Neg, syn1Neg);

        // the last window of a cbow batch holds a word that is not a row of syn0
        assertThrows(RuntimeException.class, () -> cbow(Nd4j.createFromArray(new int[]{1, 2}),
                Nd4j.createFromArray(new int[][]{{1, 2, 3}, {4, 5, VOCAB}}),
                Nd4j.createFromArray(new int[][]{{0, 0, 0}, {0, 0, 0}}), Nd4j.createFromArray(new int[]{1, 2}),
                Nd4j.createFromArray(new int[][]{{4, 5}, {2, 9}}), Nd4j.createFromArray(new int[][]{{0, 1}, {1, 0}}),
                syn0, syn1, Nd4j.empty(DataType.FLOAT), model.expArray(), Nd4j.empty(DataType.FLOAT), 0,
                Nd4j.createFromArray(new double[]{0.05, 0.04}), Nd4j.empty(DataType.FLOAT),
                Nd4j.empty(DataType.INT32), true, 1, CBOW_INFERENCE_MIN_ALPHA));
        assertTable("syn0 after the rejected cbow batch", model.syn0, syn0);
        assertTable("syn1 after the rejected cbow batch", model.syn1, syn1);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void tablesTheHelpersCannotReadAreRejected(Nd4jBackend backend) {
        Model model = new Model(45);
        INDArray syn0 = model.syn0Array();
        INDArray syn1Neg = model.syn1NegArray();

        // the negative table is read as the type of syn0
        assertThrows(RuntimeException.class, () -> skipgram(Nd4j.scalar(2), Nd4j.scalar(4),
                Nd4j.empty(DataType.INT32), Nd4j.empty(DataType.INT8), syn0, Nd4j.empty(DataType.FLOAT), syn1Neg,
                model.expArray(), negativeTable().castTo(DataType.DOUBLE), 2, Nd4j.scalar(ALPHA),
                Nd4j.empty(DataType.FLOAT), 1));

        // an inference vector as long as a row of syn0 and no longer
        assertThrows(RuntimeException.class, () -> skipgram(Nd4j.scalar(0), Nd4j.scalar(4),
                Nd4j.empty(DataType.INT32), Nd4j.empty(DataType.INT8), syn0, Nd4j.empty(DataType.FLOAT), syn1Neg,
                model.expArray(), negativeTable(), 2, Nd4j.scalar(ALPHA), Nd4j.createFromArray(new float[DIM - 1]),
                1));

        // syn1Neg has to be as wide as syn0
        assertThrows(RuntimeException.class, () -> skipgram(Nd4j.scalar(2), Nd4j.scalar(4),
                Nd4j.empty(DataType.INT32), Nd4j.empty(DataType.INT8), syn0, Nd4j.empty(DataType.FLOAT),
                Nd4j.createFromArray(new float[VOCAB][DIM - 1]), model.expArray(), negativeTable(), 2,
                Nd4j.scalar(ALPHA), Nd4j.empty(DataType.FLOAT), 1));
        assertTable("syn0 after the rejected rounds", model.syn0, syn0);
        assertTable("syn1Neg after the rejected rounds", model.syn1Neg, syn1Neg);
    }
}
