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
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Validate the output - and shape function - of a single operation.
 * <p>
 * Used with {@link OpValidation}
 *
 * @author Alex Black
 */
@Data
@Accessors(fluent = true)
public class OpTestCase {

    private final DynamicCustomOp op;
    private Map<Integer, Function<INDArray, String>> testFns = new HashMap<>();
    private Map<Integer, long[]> expShapes = new HashMap<>();

    public OpTestCase(@NonNull DynamicCustomOp op) {
        this.op = op;
    }

    /**
     * Validate the op output using INDArray.equals(INDArray)
     *
     * @param outputNum Number of the output
     * @param expected  Expected INDArray
     */
    public OpTestCase expectedOutput(int outputNum, INDArray expected) {
        testFns.put(outputNum, new EqualityFn(expected));
        expShapes.put(outputNum, expected.shape());
        return this;
    }

    /**
     * Validate the output for a single variable using element-wise relative error:
     * relError = abs(x-y)/(abs(x)+abs(y)), with x=y=0 case defined to be 0.0.
     * Also has a minimum absolute error condition, which must be satisfied for the relative error failure to be considered
     * legitimate
     *
     * @param outputNum   output number
     * @param expected    Expected INDArray
     * @param maxRelError Maximum allowable relative error
     * @param minAbsError Minimum absolute error for a failure to be considered legitimate
     */
    public OpTestCase expectedOutputRelError(int outputNum, @NonNull INDArray expected, double maxRelError, double minAbsError) {
        testFns.put(outputNum, new RelErrorFn(expected, maxRelError, minAbsError));
        expShapes.put(outputNum, expected.shape());
        return this;
    }

    /**
     * @param outputNum    Output number to check
     * @param expShape     Expected shape for the output
     * @param validationFn Function to use to validate the correctness of the specific Op. Should return null
     *                     if validation passes, or an error message if the op validation fails
     */
    public OpTestCase expectedOutput(int outputNum, @NonNull long[] expShape, @NonNull Function<INDArray, String> validationFn) {
        testFns.put(outputNum, validationFn);
        expShapes.put(outputNum, expShape);
        return this;
    }
}
