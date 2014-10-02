package org.nd4j.linalg.indexing;

import com.google.common.base.Function;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 10/1/14.
 */
public class Apply {

    private INDArray toTransform;
    private Function apply;

    public void apply() {
        if(toTransform instanceof IComplexNDArray) {
            IComplexNDArray linear = (IComplexNDArray) toTransform.linearView();
            for(int i = 0; i < linear.length(); i++) {

            }
        }
        else {
            INDArray linear = toTransform.linearView();
            for(int i = 0; i < linear.length(); i++) {

            }
        }

    }

}
