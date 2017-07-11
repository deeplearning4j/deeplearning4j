package org.deeplearning4j.nn.simple.multiclass;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@link RankClassificationResult}
 * is an abstraction over an activation matrix
 * for ranking classes.
 *
 * @author Adam Gibson
 */
@Data
public class RankClassificationResult implements Serializable {
    private int[][] rankedIndices;
    private float[][] probabilities;
    private List<String> labels;
    private List<String> maxLabels;

    /**
     * Takes in just a classification matrix
     * and initializes the labels to just be indices
     * @param outcome the outcome matrix (usually from a softmax
     *                or sigmoid output)
     */
    public RankClassificationResult(INDArray outcome) {
        this(outcome, null);
    }

    /**
     * Takes in a classification matrix
     * and the labels for each column
     * @param outcome the outcome
     * @param labels the labels for the outcomes
     */
    public RankClassificationResult(INDArray outcome, List<String> labels) {

        if (outcome.rank() > 2) {
            throw new ND4JIllegalStateException("Only works with vectors and matrices right now");
        }

        INDArray[] maxWithIndices = Nd4j.sortWithIndices(outcome, -1, false);
        INDArray indexes = maxWithIndices[0];
        //default to integers for labels
        if (labels == null) {
            this.labels = new ArrayList<>(outcome.columns());
            for (int i = 0; i < outcome.columns(); i++) {
                this.labels.add(String.valueOf(i));
            }
        } else {
            this.labels = new ArrayList<>(labels);
        }

        rankedIndices = new int[indexes.rows()][indexes.columns()];
        probabilities = new float[outcome.rows()][outcome.columns()];
        for (int i = 0; i < indexes.rows(); i++) {
            for (int j = 0; j < indexes.columns(); j++) {
                rankedIndices[i][j] = indexes.getInt(i, j);
                probabilities[i][j] = outcome.getFloat(new int[] {i, j});
            }
        }

        //initialize max outcomes
        maxOutcomes();

    }

    /**
     * Get the max index for the given row
     * @param r the row to get the max index for
     * @return the label for the given
     * element
     */
    public String maxOutcomeForRow(int r) {
        return labels.get((rankedIndices[r][0]));
    }

    public List<String> maxOutcomes() {
        if (maxLabels == null) {
            maxLabels = new ArrayList<>(rankedIndices.length);
            for (int i = 0; i < rankedIndices.length; i++) {
                maxLabels.add(maxOutcomeForRow(i));
            }

            return maxLabels;
        }

        else
            return maxLabels;
    }

}
