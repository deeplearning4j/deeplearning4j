package org.nd4j.autodiff.validation;

import lombok.Data;
import lombok.NonNull;
import lombok.experimental.Accessors;
import org.nd4j.autodiff.validation.functions.EqualityFn;
import org.nd4j.autodiff.validation.functions.RelErrorFn;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.function.Function;

import java.util.HashMap;
import java.util.Map;

@Data
@Accessors(fluent = true)
public class OpTestCase {

    private final DynamicCustomOp op;
    private Map<Integer,Function<INDArray,String>> testFns = new HashMap<>();
    private Map<Integer,long[]> expShapes = new HashMap<>();

    public OpTestCase(@NonNull DynamicCustomOp op){
        this.op = op;
    }

    public OpTestCase expectedOutput(int outputNum, INDArray expected){
        testFns.put(outputNum, new EqualityFn(expected));
        expShapes.put(outputNum, expected.shape());
        return this;
    }

    public OpTestCase expectedOutputRelError(int outputNum, @NonNull INDArray expected, double maxRelError, double minAbsError){
        testFns.put(outputNum, new RelErrorFn(expected, maxRelError, minAbsError));
        expShapes.put(outputNum, expected.shape());
        return this;
    }
}
