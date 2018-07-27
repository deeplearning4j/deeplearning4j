/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.autodiff.validation;

import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.Accessors;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.functions.EqualityFn;
import org.nd4j.autodiff.validation.functions.RelErrorFn;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.function.Function;

import java.util.*;

/**
 * TestCase: Validate a SameDiff instance.
 * Can be used to validate gradients (enabled by default) and expected outputs (forward pass for variables) if such
 * outputs are provided.
 * <p>
 * Used with {@link OpValidation}
 *
 * @author Alex Black
 */
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
    private Map<String, Function<INDArray, String>> fwdTestFns;

    //Gradient check configuration
    private boolean gradientCheck = true;
    private boolean gradCheckPrint = GC_DEFAULT_PRINT;
    private boolean gradCheckDefaultExitFirstFailure = GC_DEFAULT_EXIT_FIRST_FAILURE;
    private boolean gradCheckDebugMode = GC_DEFAULT_DEBUG_MODE;
    private double gradCheckEpsilon = GC_DEFAULT_EPS;
    private double gradCheckMaxRelativeError = GC_DEFAULT_MAX_REL_ERROR;
    private double gradCheckMinAbsError = GC_DEFAULT_MIN_ABS_ERROR;
    private Set<String> gradCheckSkipVariables;


    /**
     * @param sameDiff SameDiff instance to test. Note: All of the required inputs should already be set
     */
    public TestCase(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
    }

    /**
     * Validate the output (forward pass) for a single variable using INDArray.equals(INDArray)
     *
     * @param name     Name of the variable to check
     * @param expected Expected INDArray
     */
    public TestCase expectedOutput(@NonNull String name, @NonNull INDArray expected) {
        return expected(name, new EqualityFn(expected));
    }

    /**
     * Validate the output (forward pass) for a single variable using element-wise relative error:
     * relError = abs(x-y)/(abs(x)+abs(y)), with x=y=0 case defined to be 0.0.
     * Also has a minimum absolute error condition, which must be satisfied for the relative error failure to be considered
     * legitimate
     *
     * @param name        Name of the variable to check
     * @param expected    Expected INDArray
     * @param maxRelError Maximum allowable relative error
     * @param minAbsError Minimum absolute error for a failure to be considered legitimate
     */
    public TestCase expectedOutputRelError(@NonNull String name, @NonNull INDArray expected, double maxRelError, double minAbsError) {
        return expected(name, new RelErrorFn(expected, maxRelError, minAbsError));
    }

    /**
     * Validate the output (forward pass) for a single variable using INDArray.equals(INDArray)
     *
     * @param var    Variable to check
     * @param output Expected INDArray
     */
    public TestCase expected(@NonNull SDVariable var, @NonNull INDArray output) {
        return expected(var.getVarName(), output);
    }

    /**
     * Validate the output (forward pass) for a single variable using INDArray.equals(INDArray)
     *
     * @param name   Name of the variable to check
     * @param output Expected INDArray
     */
    public TestCase expected(@NonNull String name, @NonNull INDArray output) {
        return expectedOutput(name, output);
    }

    public TestCase expected(SDVariable var, Function<INDArray,String> validationFn){
        return expected(var.getVarName(), validationFn);
    }

    /**
     * @param name         The name of the variable to check
     * @param validationFn Function to use to validate the correctness of the specific Op. Should return null
     *                     if validation passes, or an error message if the op validation fails
     */
    public TestCase expected(String name, Function<INDArray, String> validationFn) {
        if (fwdTestFns == null)
            fwdTestFns = new LinkedHashMap<>();
        fwdTestFns.put(name, validationFn);
        return this;
    }

    public Set<String> gradCheckSkipVariables() {
        return gradCheckSkipVariables;
    }

    /**
     * Specify the input variables that should NOT be gradient checked.
     * For example, if an input is an integer index (not real valued) it should be skipped as such an input cannot
     * be gradient checked
     *
     * @param toSkip Name of the input variables to skip gradient check for
     */
    public TestCase gradCheckSkipVariables(String... toSkip) {
        if (gradCheckSkipVariables == null)
            gradCheckSkipVariables = new LinkedHashSet<>();
        Collections.addAll(gradCheckSkipVariables, toSkip);
        return this;
    }


    public void assertConfigValid() {
        Preconditions.checkNotNull(sameDiff, "SameDiff instance cannot be null%s", testNameErrMsg());
        Preconditions.checkState(gradientCheck || (fwdTestFns != null && fwdTestFns.size() > 0), "Test case is empty: nothing to test" +
                " (gradientCheck == false and no expected results available)%s", testNameErrMsg());
    }

    public String testNameErrMsg() {
        if (testName == null)
            return "";
        return " - Test name: \"" + testName + "\"";
    }

}
