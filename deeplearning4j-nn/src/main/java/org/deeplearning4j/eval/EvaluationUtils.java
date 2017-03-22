package org.deeplearning4j.eval;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Utility methods for performing evaluation
 *
 * @author Alex Black
 */
public class EvaluationUtils {

    public static INDArray reshapeTimeSeriesTo2d(INDArray labels) {
        int[] labelsShape = labels.shape();
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

        INDArray oneDMask = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(outputMask);
        float[] f = oneDMask.dup().data().asFloat();
        int[] rowsToPull = new int[f.length];
        int usedCount = 0;
        for (int i = 0; i < f.length; i++) {
            if (f[i] == 1.0f) {
                rowsToPull[usedCount++] = i;
            }
        }
        rowsToPull = Arrays.copyOfRange(rowsToPull, 0, usedCount);

        labels2d = Nd4j.pullRows(labels2d, 1, rowsToPull);
        predicted2d = Nd4j.pullRows(predicted2d, 1, rowsToPull);

        return new Pair<>(labels2d, predicted2d);
    }
}
