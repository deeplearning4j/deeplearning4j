package org.deeplearning4j.nn.simple.multiclass;

import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 4/28/17.
 */
@Data
public class MultiClassResult {
    private int[][] rankedIndices;
    private List<String> labels;

    public MultiClassResult(INDArray outcome) {
        this(outcome,null);
    }
    public MultiClassResult(INDArray outcome,List<String> labels) {

        if(outcome.rank() > 2) {
            throw new ND4JIllegalStateException("Only works with vectors and matrices right now");
        }

        INDArray[] maxWithIndices = Nd4j.sortWithIndices(outcome,-1,false);
        INDArray indexes = maxWithIndices[0];
        //default to integers for labels
        if(labels == null) {
            this.labels = new ArrayList<>(outcome.columns());
            for(int i = 0; i < outcome.columns(); i++) {
                this.labels.add(String.valueOf(i));
            }
        }
        else {
            this.labels = new ArrayList<>(labels);
        }

        rankedIndices = new int[indexes.rows()][indexes.columns()];
        for(int i = 0; i < indexes.rows(); i++) {
            for(int j = 0; j  < indexes.columns(); j++) {
                rankedIndices[i][j] = indexes.getInt(i,j);
            }
        }
    }

    /**
     * Get the max index for the given row
     * @param r
     * @return
     */
    public String maxOutcomeForRow(int r) {
        return labels.get((rankedIndices[r][0]));
    }

}
