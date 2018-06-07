package org.nd4j.autodiff.validation.functions;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.function.Function;

@AllArgsConstructor
public class EqualityFn implements Function<INDArray,String> {
    private final INDArray expected;

    @Override
    public String apply(INDArray actual) {
        if(expected.equals(actual)){
            return null;
        }
        return "INDArray equality failed";
    }
}
