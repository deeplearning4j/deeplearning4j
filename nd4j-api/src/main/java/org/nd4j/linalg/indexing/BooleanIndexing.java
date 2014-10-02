package org.nd4j.linalg.indexing;

import com.google.common.base.Function;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Boolean indexing
 * @author Adam Gibson
 */
public class BooleanIndexing {


    /**
     * Based on the matching elements
     * transform to based on condition to with function function
     * @param to the ndarray to transform
     * @param condition  the condition on transform
     * @param function the function to apply the transform to
     */
    public static void applyWhere(INDArray to,Condition condition,Function<Number,Number> function) {
        INDArray linear = to.linearView();
        for(int i = 0; i < linear.linearView().length(); i++) {
            if(condition.apply(linear.get(i))) {
                linear.putScalar(i,function.apply(linear.get(i)));
            }
        }
    }


    /**
     * Based on the matching elements
     * transform to based on condition to with function function
     * @param to the ndarray to transform
     * @param condition  the condition on transform
     * @param function the function to apply the transform to
     */
    public static void applyWhere(IComplexNDArray to,Condition condition,Function<IComplexNumber,IComplexNumber> function) {
        IComplexNDArray linear = to.linearView();
        for(int i = 0; i < linear.linearView().length(); i++) {
            if(condition.apply(linear.get(i))) {
                linear.putScalar(i,function.apply(linear.getComplex(i)));
            }
        }
    }



}
