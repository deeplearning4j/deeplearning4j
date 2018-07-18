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

package org.deeplearning4j.eval;

import lombok.val;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;

/**
 * Utility methods for performing evaluation
 *
 * @author Alex Black
 */
public class EvaluationUtils {

    /**
     * Calculate the precision from true positive and false positive counts
     *
     * @param tpCount  True positive count
     * @param fpCount  False positive count
     * @param edgeCase Edge case value use to avoid 0/0
     * @return Precision
     */
    public static double precision(long tpCount, long fpCount, double edgeCase) {
        //Edge case
        if (tpCount == 0 && fpCount == 0) {
            return edgeCase;
        }

        return tpCount / (double) (tpCount + fpCount);
    }

    /**
     * Calculate the recall from true positive and false negative counts
     *
     * @param tpCount  True positive count
     * @param fnCount  False negative count
     * @param edgeCase Edge case values used to avoid 0/0
     * @return Recall
     */
    public static double recall(long tpCount, long fnCount, double edgeCase) {
        //Edge case
        if (tpCount == 0 && fnCount == 0) {
            return edgeCase;
        }

        return tpCount / (double) (tpCount + fnCount);
    }

    /**
     * Calculate the false positive rate from the false positive count and true negative count
     *
     * @param fpCount  False positive count
     * @param tnCount  True negative count
     * @param edgeCase Edge case values are used to avoid 0/0
     * @return False positive rate
     */
    public static double falsePositiveRate(long fpCount, long tnCount, double edgeCase) {
        //Edge case
        if (fpCount == 0 && tnCount == 0) {
            return edgeCase;
        }
        return fpCount / (double) (fpCount + tnCount);
    }

    /**
     * Calculate the false negative rate from the false negative counts and true positive count
     *
     * @param fnCount  False negative count
     * @param tpCount  True positive count
     * @param edgeCase Edge case value to use to avoid 0/0
     * @return False negative rate
     */
    public static double falseNegativeRate(long fnCount, long tpCount, double edgeCase) {
        //Edge case
        if (fnCount == 0 && tpCount == 0) {
            return edgeCase;
        }

        return fnCount / (double) (fnCount + tpCount);
    }

    /**
     * Calculate the F beta value from counts
     *
     * @param beta Beta of value to use
     * @param tp   True positive count
     * @param fp   False positive count
     * @param fn   False negative count
     * @return F beta
     */
    public static double fBeta(double beta, long tp, long fp, long fn) {
        double prec = tp / ((double) tp + fp);
        double recall = tp / ((double) tp + fn);
        return fBeta(beta, prec, recall);
    }

    /**
     * Calculate the F-beta value from precision and recall
     *
     * @param beta      Beta value to use
     * @param precision Precision
     * @param recall    Recall
     * @return F-beta value
     */
    public static double fBeta(double beta, double precision, double recall) {
        if (precision == 0.0 || recall == 0.0)
            return 0;

        double numerator = (1 + beta * beta) * precision * recall;
        double denominator = beta * beta * precision + recall;

        return numerator / denominator;
    }

    /**
     * Calculate the G-measure from precision and recall
     *
     * @param precision Precision value
     * @param recall    Recall value
     * @return G-measure
     */
    public static double gMeasure(double precision, double recall) {
        return Math.sqrt(precision * recall);
    }

    /**
     * Calculate the binary Matthews correlation coefficient from counts
     *
     * @param tp True positive count
     * @param fp False positive counts
     * @param fn False negative counts
     * @param tn True negative count
     * @return Matthews correlation coefficient
     */
    public static double matthewsCorrelation(long tp, long fp, long fn, long tn) {
        double numerator = ((double) tp) * tn - ((double) fp) * fn;
        double denominator = Math.sqrt(((double) tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
        return numerator / denominator;
    }


    public static INDArray reshapeTimeSeriesTo2d(INDArray labels) {
        val labelsShape = labels.shape();
        INDArray labels2d;
        if (labelsShape[0] == 1) {
            labels2d = labels.tensorAlongDimension(0, 1, 2).permutei(1, 0); //Edge case: miniBatchSize==1
        } else if (labelsShape[2] == 1) {
            labels2d = labels.tensorAlongDimension(0, 1, 0); //Edge case: timeSeriesLength=1
        } else {
            labels2d = labels.permute(0, 2, 1);
            labels2d = labels2d.reshape('f', labelsShape[0] * labelsShape[2], labelsShape[1]);
        }
        return labels2d;
    }

    public static Pair<INDArray, INDArray> extractNonMaskedTimeSteps(INDArray labels, INDArray predicted,
                    INDArray outputMask) {
        if (labels.rank() != 3 || predicted.rank() != 3) {
            throw new IllegalArgumentException("Invalid data: expect rank 3 arrays. Got arrays with shapes labels="
                            + Arrays.toString(labels.shape()) + ", predictions=" + Arrays.toString(predicted.shape()));
        }

        //Reshaping here: basically RnnToFeedForwardPreProcessor...
        //Dup to f order, to ensure consistent buffer for reshaping
        labels = labels.dup('f');
        predicted = predicted.dup('f');

        INDArray labels2d = EvaluationUtils.reshapeTimeSeriesTo2d(labels);
        INDArray predicted2d = EvaluationUtils.reshapeTimeSeriesTo2d(predicted);

        if (outputMask == null) {
            return new Pair<>(labels2d, predicted2d);
        }

        INDArray oneDMask = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(outputMask, LayerWorkspaceMgr.noWorkspacesImmutable(), ArrayType.INPUT);
        float[] f = oneDMask.dup().data().asFloat();
        int[] rowsToPull = new int[f.length];
        int usedCount = 0;
        for (int i = 0; i < f.length; i++) {
            if (f[i] == 1.0f) {
                rowsToPull[usedCount++] = i;
            }
        }
        if(usedCount == 0){
            //Edge case: all time steps are masked -> nothing to extract
            return null;
        }
        rowsToPull = Arrays.copyOfRange(rowsToPull, 0, usedCount);

        labels2d = Nd4j.pullRows(labels2d, 1, rowsToPull);
        predicted2d = Nd4j.pullRows(predicted2d, 1, rowsToPull);

        return new Pair<>(labels2d, predicted2d);
    }
}
