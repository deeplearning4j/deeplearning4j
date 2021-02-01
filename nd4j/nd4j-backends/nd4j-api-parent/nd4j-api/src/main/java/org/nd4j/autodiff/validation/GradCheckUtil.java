/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.validation;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.validation.listeners.NonInplaceValidationListener;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Field;
import java.util.*;

/**
 * Gradient check utility
 *
 * @author Adam Gibson
 */
@Slf4j
public class GradCheckUtil {

    public enum Subset {EVERY_N, RANDOM}

    public static final boolean DEFAULT_PRINT = false;
    public static final boolean DEFAULT_EXIT_FIRST_FAILURE = false;
    public static final boolean DEFAULT_DEBUG_MODE = false;
    public static final double DEFAULT_EPS = 1e-5;
    public static final double DEFAULT_MAX_REL_ERROR = 1e-5;
    public static final double DEFAULT_MIN_ABS_ERROR = 1e-6;

    public static boolean checkGradients(TestCase t){
        return checkGradients(t.sameDiff(), t.placeholderValues(), t.gradCheckEpsilon(), t.gradCheckMaxRelativeError(), t.gradCheckMinAbsError(),
                t.gradCheckPrint(), t.gradCheckDefaultExitFirstFailure(), false, t.gradCheckDebugMode(), t.gradCheckSkipVariables(), t.gradCheckMask());
    }

    public static boolean checkGradients(SameDiff sd, Map<String,INDArray> placeholderValues, String... skipVariables){
        Set<String> skip = null;
        if(skipVariables != null){
            skip = new HashSet<>();
            Collections.addAll(skip, skipVariables);
        }
        return checkGradients(sd, placeholderValues, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR, DEFAULT_PRINT, DEFAULT_EXIT_FIRST_FAILURE,
                false, DEFAULT_DEBUG_MODE, skip, null);
    }

    public static boolean checkGradients(SameDiff sd, Map<String,INDArray> placeholderValues, boolean print, boolean exitOnFirstFailure){
        return checkGradients(sd, placeholderValues, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR, DEFAULT_MIN_ABS_ERROR, print, exitOnFirstFailure);
    }


    public static boolean checkGradients(SameDiff sd, Map<String,INDArray> placeholderValues, double eps, double maxRelError, double minAbsError, boolean print,
                                         boolean exitOnFirstFailure) {
        return checkGradients(sd, placeholderValues, eps, maxRelError, minAbsError, print, exitOnFirstFailure, false, DEFAULT_DEBUG_MODE, null, null);
    }

    public static boolean checkGradients(SameDiff sd, Map<String,INDArray> placeholderValues, double eps, double maxRelError, double minAbsError, boolean print,
                                         boolean exitOnFirstFailure, boolean skipValidation, boolean debugMode, Set<String> skipVariables, Map<String,INDArray> gradCheckMask) {
        return checkGradients(sd, placeholderValues, eps, maxRelError, minAbsError, print, exitOnFirstFailure, skipValidation, debugMode,
                skipVariables, gradCheckMask, -1, null);
    }

    public static boolean checkGradients(SameDiff sd, Map<String,INDArray> placeholderValues, double eps, double maxRelError, double minAbsError, boolean print,
                                         boolean exitOnFirstFailure, boolean skipValidation, boolean debugMode, Set<String> skipVariables, Map<String,INDArray> gradCheckMask,
                                         int maxPerParam, Subset subset){

        boolean debugBefore = sd.isDebugMode();
        if(debugMode){
            sd.enableDebugMode();
        }

        //Validation sanity checks:
        if(!skipValidation){
            validateInternalState(sd, true);
        }

        //Check data type:
        if(Nd4j.dataType() != DataType.DOUBLE){
            throw new IllegalStateException("Data type must be set to double");
        }

        Set<String> fnOutputs = new HashSet<>();
        for(DifferentialFunction f : sd.ops()){
            for(SDVariable s : f.outputVariables()){
                fnOutputs.add(s.name());
            }
        }

        //Check that all non-Array type SDVariables have arrays associated with them
        for(Variable v : sd.getVariables().values()){
            if(v.getVariable().getVariableType() == VariableType.ARRAY){
                //OK if variable is not available for this, it'll be created during forward pass
                continue;
            }

            if(v.getVariable().getArr(true) == null){
                throw new IllegalStateException("Variable \"" + v.getName() + "\" does not have array associated with it");
            }
        }

        //Do forward pass, check that output is a scalar:
        List<String> lossFnVariables = sd.getLossVariables();
        Preconditions.checkState(lossFnVariables != null && !lossFnVariables.isEmpty(), "Expected 1 or more loss function variables for gradient check, got %s", lossFnVariables);

        //TODO also check that all inputs are non-zero (otherwise: consider out = sum(x * y) with all x and y being 0
        // in this case, gradients of x and y are all 0 too

        //Collect variables to get gradients for - we want placeholders AND variables
        Set<String> varsNeedingGrads = new HashSet<>();
        for(Variable v : sd.getVariables().values()){
            if(v.getVariable().dataType().isFPType() && (v.getVariable().getVariableType() == VariableType.VARIABLE || v.getVariable().getVariableType() == VariableType.PLACEHOLDER)){
                SDVariable g = v.getVariable().getGradient();
                Preconditions.checkNotNull(g, "No gradient variable found for variable %s", v.getVariable());
                varsNeedingGrads.add(v.getName());
            }
        }

        //Add non-inplace validation listener, to check that non-inplace ops don't modify their inputs
        List<Listener> listenersBefore = new ArrayList<>(sd.getListeners());
        int listenerIdx = -1;
        if(listenersBefore.isEmpty()){
            sd.addListeners(new NonInplaceValidationListener());
            listenerIdx = 0;
        } else {
            boolean found = false;
            int i=0;
            for(Listener l : listenersBefore){
                if(l instanceof NonInplaceValidationListener){
                    found = true;
                    listenerIdx = i;
                    break;
                }
                i++;
            }
            if(!found){
                sd.addListeners(new NonInplaceValidationListener());
                listenerIdx = i;
            }
        }


        Map<String,INDArray> gm = sd.calculateGradients(placeholderValues, varsNeedingGrads);

        //Remove listener, to reduce overhead
        sd.getListeners().remove(listenerIdx);

        Map<String,INDArray> grad = new HashMap<>();
        for(SDVariable v : sd.variables()){
            if (fnOutputs.contains(v.name())) {
                //This is not an input to the graph
                continue;
            }
            if(!v.hasGradient()){
                //Skip non-fp variables, or variables that don't impact loss function value
                continue;
            }
            SDVariable g = sd.grad(v.name());
            if(g == null){
                throw new IllegalStateException("Null gradient variable for \"" + v.name() + "\"");
            }
            INDArray ga = gm.get(v.name());
            if(ga == null){
                throw new IllegalStateException("Null gradient array encountered for variable: " + v.name());
            }
            if(!Arrays.equals(v.getArr().shape(), ga.shape())){
                throw new IllegalStateException("Gradient shape does not match variable shape for variable \"" +
                    v.name() + "\": shape " + Arrays.toString(v.getArr().shape()) + " vs. gradient shape " +
                    Arrays.toString(ga.shape()));
            }
            grad.put(v.name(), ga.dup());
        }

        //Validate gradients for each variable:
        int totalNFailures = 0;
        int totalCount = 0;
        double maxError = 0.0;
        Random r = new Random(12345);
        for(SDVariable s : sd.variables()){
            if (fnOutputs.contains(s.name()) || !s.dataType().isFPType()) {
                //This is not an input to the graph, or is not a floating point input (so can't be gradient checked)
                continue;
            }

            if(skipVariables != null && skipVariables.contains(s.name())){
                log.info("Grad check: skipping variable \"{}\"", s.name());
                continue;
            }

            if(s.dataType() != DataType.DOUBLE){
                log.warn("DataType for variable {} is not double (is: {}) may cause precision issues in gradient checks", s.name(), s.dataType());
            }

            String name = s.name();
            INDArray a = s.getArr();
            long n = a.length();
            if(print){
                log.info("Starting test for variable \"{}\" with {} values", s.name(), n);
            }

            Iterator<long[]> iter;
            if(maxPerParam > 0 && subset != null && maxPerParam < a.length()){
                //Subset case
                long[] shape = a.shape();
                List<long[]> l = new ArrayList<>();
                if(subset == Subset.RANDOM){
                    Set<Integer> set = new HashSet<>();
                    while(set.size() < maxPerParam){
                        int next = r.nextInt((int)a.length());
                        set.add(next);
                    }
                    List<Integer> sorted = new ArrayList<>(set);
                    Collections.sort(sorted);

                    for(Integer i : sorted){
                        long[] pos = Shape.ind2subC(shape, i);
                        l.add(pos);
                    }
                } else {
                    //Every N
                    long everyN = n / maxPerParam;
                    long curr = 0;
                    while(curr < n){
                        long[] pos = Shape.ind2subC(shape, curr);
                        l.add(pos);
                        curr += everyN;
                    }
                }
                iter = l.iterator();
            } else {
                //Standard case: do all parameters
                iter = new NdIndexIterator('c',a.shape());
            }

            INDArray varMask = (gradCheckMask == null ? null : gradCheckMask.get(s.name()));

            if(varMask != null){
                Preconditions.checkState(a.equalShapes(varMask), "Variable \"%s\": Gradient check mask and array shapes must be equal: got %s vs. mask shape %s", s.name(), a.shape(), varMask.shape());
                Preconditions.checkState(varMask.dataType() == DataType.BOOL, "Variable \"%s\": Gradient check mask must be BOOLEAN datatype, got %s", s.name(), varMask.dataType());
            }

            int i=0;
            while(iter.hasNext()){
                long[] idx = iter.next();
                String strIdx = null;
                if(print){
                    strIdx = Arrays.toString(idx).replaceAll(" ","");
                }

                boolean maskValue = (varMask == null || (varMask.getDouble(idx) != 0));
                if(!maskValue){
                    //Skip this specific entry (masked out)
                    continue;
                }

                totalCount++;
                double orig = a.getDouble(idx);
                a.putScalar(idx, orig+eps);
                double scorePlus = 0.0;
                Map<String,INDArray> m = sd.output(placeholderValues, lossFnVariables);//.get(outName).sumNumber().doubleValue();
                for(INDArray arr : m.values()){
                    scorePlus += arr.sumNumber().doubleValue();
                }
                a.putScalar(idx, orig-eps);
                m = sd.output(placeholderValues, lossFnVariables);
                double scoreMinus = 0.0;
                for(INDArray arr : m.values()){
                    scoreMinus += arr.sumNumber().doubleValue();
                }
                a.putScalar(idx, orig);

                double numericalGrad = (scorePlus - scoreMinus) / (2 * eps);
                INDArray aGrad = grad.get(s.name());
                if(aGrad == null){
                    log.warn("No gradient array for variable \"{}\" was found, skipping variable...", s.name());
                    continue;
                }
                double analyticGrad = aGrad.getDouble(idx);

                if (Double.isInfinite(numericalGrad) || Double.isNaN(numericalGrad)) {
                    throw new IllegalStateException("Numerical gradient was " + numericalGrad + " for variable \"" + name
                            + "\", parameter " + i + " of " + n + " (position: " + strIdx + ")");
                }
                if (Double.isInfinite(analyticGrad) || Double.isNaN(analyticGrad)) {
                    throw new IllegalStateException("Analytic (SameDiff) gradient was " + analyticGrad + " for variable \"" + name
                            + "\", parameter " + i + " of " + n + " (position: " + strIdx + ")");
                }


                double relError;
                if(numericalGrad == 0.0 && analyticGrad == 0.0){
                    relError = 0.0;
                } else {
                    relError = Math.abs(analyticGrad - numericalGrad) / (Math.abs(Math.abs(analyticGrad) + Math.abs(numericalGrad)));
                }

                if (relError > maxError)
                    maxError = relError;

                if (relError > maxRelError || Double.isNaN(relError)) {
                    double absError = Math.abs(analyticGrad - numericalGrad);
                    if (absError < minAbsError) {
                        if(print) {
                            log.info("Param " + i + " (" + name + strIdx + ") passed: grad= " + analyticGrad
                                    + ", numericalGrad= " + numericalGrad + ", relError= " + relError
                                    + "; absolute error = " + absError + " < minAbsoluteError = " + minAbsError);
                        }
                    } else {
                        log.info("Param " + i + " (" + name + strIdx + ") FAILED: grad= " + analyticGrad
                                + ", numericalGrad= " + numericalGrad + ", relError= " + relError
                                + ", absError=" + absError
                                + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                        if (exitOnFirstFailure)
                            return false;
                        totalNFailures++;
                    }
                } else if (print) {
                    log.info("Param " + i + " (" + name + strIdx + ") passed: grad= " + analyticGrad + ", numericalGrad= "
                            + numericalGrad + ", relError= " + relError);
                }
                i++;
            }
        }

        int nPass = totalCount - totalNFailures;
        log.info("GradCheckUtil.checkGradients(): " + totalCount + " params checked, " + nPass + " passed, "
                + totalNFailures + " failed. Largest relative error = " + maxError);

        if(debugMode && !debugBefore){
            sd.disableDebugging();
        }

        return totalNFailures == 0;
    }


    /**
     * Gradient check the ACTIVATIONS (i.e., ARRAY type SDVariables) as opposed to the parameters of a network (as
     * are tested in {@link #checkGradients(SameDiff, Map, double, double, double, boolean, boolean, boolean, boolean, Set, Map, int, Subset)}
     * @param config Configuration for gradient check
     * @return True if gradient checks pass
     */
    public static boolean checkActivationGradients(ActGradConfig config){
        SameDiff sd = config.getSd();
        List<String> actGrads = config.getActivationGradsToCheck();
        double maxRelError = config.getMaxRelError();
        double minAbsError = config.getMinAbsError();

        Preconditions.checkState(sd != null, "SameDiff instance was not set in configuration");
        Preconditions.checkState(actGrads != null && !actGrads.isEmpty(), "No activation gradients were specified to gradient check");
        Preconditions.checkState(config.getEps() > 0.0, "Epsilon has not been set");
        Preconditions.checkState(maxRelError > 0.0, "Max relative error must be set (is 0.0)");

        for(String s : actGrads){
            SDVariable v = sd.getVariables().get(s).getVariable();
            Preconditions.checkState(v != null, "No variable with name \"%s\" was found", s);
            Preconditions.checkState(v.getVariableType() == VariableType.ARRAY, "Only variables with type ARRAY may be " +
                    "gradient checked using this method. Variable \"%s\" has type %s", s, v.getVariableType());
            Preconditions.checkState(v.dataType().isFPType(), "Cannot gradient check activation variable \"%s\": must be floating point type. Is type: %s", s, v.dataType());
            if(v.dataType() != DataType.DOUBLE){
                log.warn("Floating point variable {} is not double precision - this may result in spurious failures due to limited precision. Variable is type: {}", s, v.dataType());
            }
        }

        boolean debugBefore = sd.isDebugMode();
        if(config.isDebugMode()){
            sd.enableDebugMode();
        }

        //Validation sanity checks:
        if(!config.isSkipValidation()){
            validateInternalState(sd, true);
        }

        //Loss function variables
        List<String> lossFnVariables = sd.getLossVariables();
        Preconditions.checkState(lossFnVariables != null && !lossFnVariables.isEmpty(), "Expected 1 or more loss function variables for gradient check, got %s", lossFnVariables);

        //TODO also check that all inputs are non-zero (otherwise: consider out = sum(x * y) with all x and y being 0
        // in this case, gradients of x and y are all 0 too

        //Collect names of variables to get gradients for - i.e., the names of the GRADIENT variables for the specified activations
        sd.createGradFunction();
        Set<String> varsRequiringGrads = new HashSet<>();
        for(String s : actGrads){
            SDVariable grad = sd.getVariable(s).gradient();
            Preconditions.checkState( grad != null,"Could not get gradient for activation \"%s\": gradient variable is null", s);
            varsRequiringGrads.add(s);
        }

        //Calculate analytical gradients
        Map<String,INDArray> grads = sd.calculateGradients(config.getPlaceholderValues(), new ArrayList<>(varsRequiringGrads));
        Map<String,INDArray> gradientsForAct = new HashMap<>();
        for(String s : actGrads){
            INDArray arr = grads.get(s);
            Preconditions.checkState(arr != null, "No activation gradient array for variable \"%s\"", s);
            gradientsForAct.put(s, arr.dup());
        }


        //Now, check gradients
        int totalNFailures = 0;
        int totalCount = 0;
        double maxError = 0.0;
        ActivationGradientCheckListener listener = new ActivationGradientCheckListener();
        sd.setListeners(listener);
        Random r = new Random(12345);
        int maxPerParam = config.getMaxPerParam();
        for(String s : actGrads){

            long n = gradientsForAct.get(s).length();
            if(config.isPrint()){
                log.info("Starting test for variable \"{}\" with {} values", s, n);
            }

            Iterator<long[]> iter;
            if(maxPerParam > 0 && config.getSubset() != null && maxPerParam < n){
                //Subset case
                long[] shape = gradientsForAct.get(s).shape();
                List<long[]> l = new ArrayList<>();
                if(config.getSubset() == Subset.RANDOM){
                    Set<Integer> set = new HashSet<>();
                    while(set.size() < maxPerParam){
                        int next = r.nextInt((int)n);
                        set.add(next);
                    }
                    List<Integer> sorted = new ArrayList<>(set);
                    Collections.sort(sorted);

                    for(Integer i : sorted){
                        long[] pos = Shape.ind2subC(shape, i);
                        l.add(pos);
                    }
                } else {
                    //Every N
                    long everyN = n / maxPerParam;
                    long curr = 0;
                    while(curr < n){
                        long[] pos = Shape.ind2subC(shape, curr);
                        l.add(pos);
                        curr += everyN;
                    }
                }
                iter = l.iterator();
            } else {
                //Standard case: do all parameters
                iter = new NdIndexIterator('c',gradientsForAct.get(s).shape());
            }

            INDArray varMask = (config.getGradCheckMask() == null ? null : config.getGradCheckMask().get(s));

            listener.setVariableName(s);

            int i=0;
            while(iter.hasNext()){
                long[] idx = iter.next();

                String strIdx = null;
                if(config.isPrint()){
                    strIdx = Arrays.toString(idx).replaceAll(" ","");
                }

                boolean maskValue = (varMask == null || (varMask.getDouble(idx) != 0));
                if(!maskValue){
                    //Skip this specific entry (masked out)
                    continue;
                }

                //Set listener to apply eps, then do forward pass:
                listener.setIdx(idx);
                listener.setEps(config.getEps());
                double scorePlus = 0.0;
                Map<String,INDArray> m = sd.output(config.getPlaceholderValues(), lossFnVariables);
                for(INDArray arr : m.values()){
                    scorePlus += arr.sumNumber().doubleValue();
                }
                listener.setEps(-config.getEps());
                m = sd.output(config.getPlaceholderValues(), lossFnVariables);
                double scoreMinus = 0.0;
                for(INDArray arr : m.values()){
                    scoreMinus += arr.sumNumber().doubleValue();
                }

                double numericalGrad = (scorePlus - scoreMinus) / (2 * config.getEps());
                double analyticGrad = gradientsForAct.get(s).getDouble(idx);

                if (Double.isInfinite(numericalGrad) || Double.isNaN(numericalGrad)) {
                    throw new IllegalStateException("Numerical gradient was " + numericalGrad + " for variable \"" + s
                            + "\", parameter " + i + " of " + n + " (position: " + strIdx + ")");
                }
                if (Double.isInfinite(analyticGrad) || Double.isNaN(analyticGrad)) {
                    throw new IllegalStateException("Analytic (SameDiff) gradient was " + analyticGrad + " for variable \"" + s
                            + "\", parameter " + i + " of " + n + " (position: " + strIdx + ")");
                }

                double relError;
                if(numericalGrad == 0.0 && analyticGrad == 0.0){
                    relError = 0.0;
                } else {
                    relError = Math.abs(analyticGrad - numericalGrad) / (Math.abs(Math.abs(analyticGrad) + Math.abs(numericalGrad)));
                }

                if (relError > maxError)
                    maxError = relError;

                if (relError > maxRelError || Double.isNaN(relError)) {
                    double absError = Math.abs(analyticGrad - numericalGrad);
                    if (absError < minAbsError) {
                        if(config.isPrint()) {
                            log.info("Param " + i + " (" + s + strIdx + ") passed: grad= " + analyticGrad
                                    + ", numericalGrad= " + numericalGrad + ", relError= " + relError
                                    + "; absolute error = " + absError + " < minAbsoluteError = " + minAbsError);
                        }
                    } else {
                        if (config.isPrint())
                            log.info("Param " + i + " (" + s + strIdx + ") FAILED: grad= " + analyticGrad
                                    + ", numericalGrad= " + numericalGrad + ", relError= " + relError
                                    + ", absError=" + absError
                                    + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                        if (config.isExitOnFirstFailure())
                            return false;
                        totalNFailures++;
                    }
                } else if (config.isPrint()) {
                    log.info("Param " + i + " (" + s + strIdx + ") passed: grad= " + analyticGrad + ", numericalGrad= "
                            + numericalGrad + ", relError= " + relError);
                }
                i++;

            }
        }

        return totalNFailures == 0;
    }

    @Builder
    @Data
    public static class ActGradConfig {
        private SameDiff sd;
        private Map<String,INDArray> placeholderValues;
        private List<String> activationGradsToCheck;
        @Builder.Default private double eps = DEFAULT_EPS;
        @Builder.Default private double maxRelError = DEFAULT_MAX_REL_ERROR;
        @Builder.Default private double minAbsError = DEFAULT_MIN_ABS_ERROR;
        @Builder.Default private boolean print = DEFAULT_PRINT;
        @Builder.Default boolean exitOnFirstFailure = DEFAULT_EXIT_FIRST_FAILURE;
        @Builder.Default private boolean skipValidation = false;
        @Builder.Default private boolean debugMode = DEFAULT_DEBUG_MODE;
        private Set<String> skipVariables;
        private Map<String,INDArray> gradCheckMask;
        int maxPerParam;
        private Subset subset;
    }


    public static void validateInternalState(SameDiff sd, boolean generateAndCheckGradFn){

        /*
        Some conditions that should always hold:
        1. incomingArgsReverse and outgoingArgsReverse:
            (a) all differential functions should be present here exactly once
            (b) The values should be valid variable names
        2. variableMap: should contain all variables, and only all variables
        3. functionArgsFor should contain all variables, all functions... same for functionOutputsFor
        4. Gradient function: should contain all of the existing functions, and more
         */

        DifferentialFunction[] dfs = sd.ops();
        List<SDVariable> vars = sd.variables();

        Set<String> varSetStr = new HashSet<>();
        for(SDVariable v : vars){
            if(varSetStr.contains(v.name())){
                throw new IllegalStateException("Variable with name " + v.name() + " already encountered");
            }
            varSetStr.add(v.name());
        }
        Preconditions.checkState(vars.size() == varSetStr.size(), "Duplicate variables in variables() list");

        //1. Check incomingArgsReverse and outgoingArgsReverse
        Map<String,SameDiffOp> ops = sd.getOps();
        Preconditions.checkState(dfs.length == ops.size(), "All functions not present in incomingArgsReverse");
        for(DifferentialFunction df : dfs){
            Preconditions.checkState(ops.containsKey(df.getOwnName()), df.getOwnName() + " not present in ops map");

            List<String> str = ops.get(df.getOwnName()).getInputsToOp();
            if(str != null) {
                for (String s : str) {
                    Preconditions.checkState(varSetStr.contains(s), "Variable " + s + " in op inputs not a known variable name");
                }
            }

            str = ops.get(df.getOwnName()).getOutputsOfOp();
            if(str != null) {
                for (String s : str) {
                    Preconditions.checkState(varSetStr.contains(s), "Variable " + s + " in op outputs not a known variable name");
                }
            }
        }

        //Also check that outgoingArgsReverse values are unique: i.e., shouldn't have the same op appearing multiple times
        Map<String,String> seen = new HashMap<>();
        for(Map.Entry<String,SameDiffOp> e : ops.entrySet()){
            List<String> varNames = e.getValue().getOutputsOfOp();
            if(varNames != null) {
                for (String s : varNames) {
                    if (seen.containsKey(s)) {
                        throw new IllegalStateException("Already saw variable \"" + s + "\" as output for op \"" + seen.get(s)
                                + "\": expected variables to be present as an output only once; also seen as output for op \"" +
                                e.getKey() + "\"");
                    }
                    seen.put(s, e.getKey());
                }
            }
        }

        //2. Check variableMap
        Map<String, Variable> variableMap = sd.getVariables();
        Preconditions.checkState(vars.size() == variableMap.size(), "Variable map size check failed");
        for(Map.Entry<String, Variable> e : variableMap.entrySet()){
            Preconditions.checkState(e.getKey().equals(e.getValue().getVariable().name()), "Name not equal");
        }

        if(generateAndCheckGradFn) {
            //3. Check gradient function
            if(sd.getFunction("grad") == null){
                sd.createGradFunction();
            }

            SameDiff gradFn = sd.getFunction("grad");
            //Run same validation for gradient fn...
            validateInternalState(gradFn, false);

            //Check that all original functions are present in the gradient function
            for(DifferentialFunction dfOrig : dfs){
                Preconditions.checkNotNull(gradFn.getOpById(dfOrig.getOwnName()), "DifferentialFunction " + dfOrig.getOwnName()
                        + " from original SameDiff instance not present in grad fn");
            }
        }
    }

    private static <T> T getObject(String fieldName, Object from, Class<?> fromClass){
        try {
            Field f = fromClass.getDeclaredField(fieldName);
            f.setAccessible(true);
            return (T)f.get(from);
        } catch (Exception e){
            throw new RuntimeException(e);
        }
    }
}
