package org.deeplearning4j.rl4j.space;

import lombok.Value;
import org.json.JSONArray;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/26/16.
 */
@Value
public class HighLowDiscrete extends DiscreteSpace {

    //size of the space also defined as the number of different actions
    INDArray matrix;

    public HighLowDiscrete(INDArray matrix) {
        super(matrix.rows());
        this.matrix = matrix;
    }

    @Override
    public Object encode(Integer a) {
        JSONArray jsonArray = new JSONArray();
        for (int i = 0; i < size; i++) {
            jsonArray.put(matrix.getDouble(i, 0));
        }
        jsonArray.put(a - 1, matrix.getDouble(a - 1, 1));
        return jsonArray;
    }

}
