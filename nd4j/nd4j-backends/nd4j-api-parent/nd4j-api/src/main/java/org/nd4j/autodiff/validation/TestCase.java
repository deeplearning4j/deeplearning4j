package org.nd4j.autodiff.validation;

import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.Accessors;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.functions.EqualityFn;
import org.nd4j.autodiff.validation.functions.RelErrorFn;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.function.Function;

import java.util.*;

@Data
@Accessors(fluent = true)
@Getter
public class TestCase {

    public static final boolean GC_DEFAULT_PRINT = true;
    public static final boolean GC_DEFAULT_EXIT_FIRST_FAILURE = false;
    public static final boolean GC_DEFAULT_DEBUG_MODE = false;
    public static final double GC_DEFAULT_EPS = 1e-5;
    public static final double GC_DEFAULT_MAX_REL_ERROR = 1e-5;
    public static final double GC_DEFAULT_MIN_ABS_ERROR = 1e-6;

    //To test
    private SameDiff sameDiff;
    private String testName;

    //Forward pass test configuration
    /*
     * Note: These forward pass functions are used to validate the output of forward pass for inputs already set
     * on the SameDiff instance.
     * Key:     The name of the variable to check the forward pass output for
     * Value:   A function to check the correctness of the output
     * NOTE: The Function<INDArray,String> should return null on correct results, and an error message otherwise
     */
    private Map<String,Function<INDArray,String>> fwdTestFns;

    //Gradient check configuration
    private boolean gradientCheck = true;
    private boolean gradCheckPrint = GC_DEFAULT_PRINT;
    private boolean gradCheckDefaultExitFirstFailure = GC_DEFAULT_EXIT_FIRST_FAILURE;
    private boolean gradCheckDebugMode = GC_DEFAULT_DEBUG_MODE;
    private double gradCheckEpsilon = GC_DEFAULT_EPS;
    private double gradCheckMaxRelativeError = GC_DEFAULT_MAX_REL_ERROR;
    private double gradCheckMinAbsError = GC_DEFAULT_MIN_ABS_ERROR;
    private Set<String> gradCheckSkipVariables;


    public TestCase(SameDiff sameDiff){
        this.sameDiff = sameDiff;
    }

    public TestCase expectedOutput(@NonNull String name, @NonNull INDArray expected){
        if(fwdTestFns == null)
            fwdTestFns = new LinkedHashMap<>();
        fwdTestFns.put(name, new EqualityFn(expected));
        return this;
    }

    public TestCase expectedOutputRelError(@NonNull String name, @NonNull INDArray expected, double maxRelError, double minAbsError){
        if(fwdTestFns == null)
            fwdTestFns = new LinkedHashMap<>();
        fwdTestFns.put(name, new RelErrorFn(expected, maxRelError, minAbsError));
        return this;
    }

    public Set<String> gradCheckSkipVariables(){
        return gradCheckSkipVariables;
    }

    public TestCase gradCheckSkipVariables(String... toSkip){
        if(gradCheckSkipVariables == null)
            gradCheckSkipVariables = new LinkedHashSet<>();
        Collections.addAll(gradCheckSkipVariables, toSkip);
        return this;
    }


    public void assertConfigValid(){
        Preconditions.checkNotNull(sameDiff, "SameDiff instance cannot be null%s", testNameErrMsg());
        Preconditions.checkState(gradientCheck || (fwdTestFns != null && fwdTestFns.size() > 0), "Test case is empty: nothing to test" +
                " (gradientCheck == false and no expected results available)%s", testNameErrMsg());
    }

    public String testNameErrMsg(){
        if(testName == null)
            return "";
        return " - Test name: \"" + testName + "\"";
    }

}
