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

import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.custom.ArgMin;
import org.nd4j.linalg.api.ops.impl.reduce.HashCode;
import org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinearDerivative;
import org.nd4j.shade.guava.collect.ImmutableSet;
import org.nd4j.shade.guava.reflect.ClassPath;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.validation.listeners.NonInplaceValidationListener;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.tensorflow.TensorflowDescriptorParser;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOpDescriptor;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.custom.BarnesEdgeForces;
import org.nd4j.linalg.api.ops.custom.BarnesHutGains;
import org.nd4j.linalg.api.ops.custom.BarnesHutSymmetrize;
import org.nd4j.linalg.api.ops.custom.SpTreeCell;
import org.nd4j.linalg.api.ops.impl.broadcast.bool.*;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.loss.bp.*;
import org.nd4j.linalg.api.ops.impl.meta.InvertedPredicateMetaOp;
import org.nd4j.linalg.api.ops.impl.meta.PostulateMetaOp;
import org.nd4j.linalg.api.ops.impl.meta.PredicateMetaOp;
import org.nd4j.linalg.api.ops.impl.meta.ReduceMetaOp;
import org.nd4j.linalg.api.ops.impl.nlp.CbowRound;
import org.nd4j.linalg.api.ops.impl.nlp.SkipGramRound;
import org.nd4j.linalg.api.ops.impl.reduce.MmulBp;
import org.nd4j.linalg.api.ops.impl.reduce.bool.All;
import org.nd4j.linalg.api.ops.impl.reduce.bool.Any;
import org.nd4j.linalg.api.ops.impl.reduce.bool.IsInf;
import org.nd4j.linalg.api.ops.impl.reduce.bool.IsNaN;
import org.nd4j.linalg.api.ops.impl.reduce.longer.MatchCondition;
import org.nd4j.linalg.api.ops.impl.reduce3.EqualsWithEps;
import org.nd4j.linalg.api.ops.impl.reduce.NormalizeMoments;
import org.nd4j.linalg.api.ops.impl.reduce.bp.*;
import org.nd4j.linalg.api.ops.impl.broadcast.*;
import org.nd4j.linalg.api.ops.impl.grid.FreeGridOp;
import org.nd4j.linalg.api.ops.impl.indexaccum.*;
import org.nd4j.linalg.api.ops.impl.layers.convolution.*;
import org.nd4j.linalg.api.ops.impl.scalar.PowDerivative;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarRemainder;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarSetValue;
import org.nd4j.linalg.api.ops.impl.shape.*;
import org.nd4j.linalg.api.ops.impl.shape.bp.ConcatBp;
import org.nd4j.linalg.api.ops.impl.shape.bp.SliceBp;
import org.nd4j.linalg.api.ops.impl.shape.bp.StridedSliceBp;
import org.nd4j.linalg.api.ops.impl.shape.bp.TileBp;
import org.nd4j.linalg.api.ops.impl.transforms.Assert;
import org.nd4j.linalg.api.ops.impl.transforms.Histogram;
import org.nd4j.linalg.api.ops.impl.transforms.bool.BooleanNot;
import org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform;
import org.nd4j.linalg.api.ops.impl.transforms.custom.*;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.BinaryMinimalRelativeError;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.TanhDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.bool.Not;
import org.nd4j.linalg.api.ops.impl.transforms.segment.bp.*;
import org.nd4j.linalg.api.ops.impl.transforms.strict.GELUDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.PreciseGELUDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SwishDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.strict.TanDerivative;
import org.nd4j.linalg.api.ops.persistence.RestoreV2;
import org.nd4j.linalg.api.ops.persistence.SaveV2;
import org.nd4j.linalg.api.ops.random.compat.RandomStandardNormal;
import org.nd4j.linalg.api.ops.random.custom.DistributionUniform;
import org.nd4j.linalg.api.ops.random.impl.*;
import org.nd4j.linalg.api.ops.random.impl.Linspace;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.primitives.Pair;
import org.tensorflow.framework.OpDef;

import java.io.IOException;
import java.lang.reflect.Modifier;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.Set;

/**
 * Main test case runner for validating ops used in SameDiff.<br>
 * This OpValidation class also collects test coverage information, to determine the op test coverage, for both
 * op outputs and gradients/backprop.
 * <br><br>
 * Two types of test cases are supported:<br>
 * 1. {@link TestCase} - Can be used to check op outputs, as well as gradients<br>
 * 2. {@link OpTestCase} - Used to check the output(s) of a single op only<br>
 * <br>
 * NOTE: For the op coverage information to work properly for ND4J tests, we need the op validation to be run as part of
 * the OpValidationSuite test suite.  * Otherwise, we could report coverage information before all test have run -
 * underestimating coverage.<br>
 * <br>
 * SINGLE OP TEST: OpValidation.validate(new OpTestCase(op).expectedOutputs(0, <INDArray here>))
 * - OpTestCase checks the output values of a single op, no backprop/gradients<br>
 * - Returns an error message if op failed, or NULL if op passed<br>
 * SAMEDIFF TEST:  OpValidation.validate(new TestCase(sameDiff).gradientCheck(true).expectedOutput("someVar", <INDArray>))<br>
 * - These tests can be used to check both gradients AND expected output, collecting coverage as required<br>
 * - Returns an error message if op failed, or NULL if op passed<br>
 * - Note gradientCheck(true) is the default<br>
 * - Expected outputs are optional<br>
 * - You can specify a function for validating the correctness of each output using {@link org.nd4j.autodiff.validation.TestCase#expected(String, Function)}<br>
 *
 * @author Alex Black
 */
@Slf4j
public class OpValidation {

    /**
     * Run test case
     *
     * @param testCase Test case to run
     * @return NULL if test passes, or error message otherwise
     */
    public static String validate(TestCase testCase) {
        return validate(testCase, false);
    }

    public static String validate(TestCase testCase, boolean exceptionsAsErrorMsg) {
        try {
            return validateHelper(testCase);
        } catch (Throwable t) {
            if (exceptionsAsErrorMsg) {
                log.info("Exception encountered - returning as error message", t);
                return "EXCEPTION: " + t.getMessage();
            }
            throw t;
        }
    }

    private static String validateHelper(TestCase testCase) {
        testCase.assertConfigValid();

        //First: collect coverage information
        collectCoverageInformation(testCase);

        //Check serialization
        ByteBuffer serializedBeforeExec = null;
        if(testCase.testFlatBufferSerialization() == TestCase.TestSerialization.BEFORE_EXEC || testCase.testFlatBufferSerialization() == TestCase.TestSerialization.BOTH){
            serializedBeforeExec = testCase.sameDiff().asFlatBuffers(true);
            Preconditions.checkNotNull(serializedBeforeExec, "Serialization failed? Null output");
        }

        SameDiff sameDiff = testCase.sameDiff();
        List<Listener> listeners = sameDiff.getListeners();
        if(listeners.isEmpty()){
            sameDiff.addListeners(new NonInplaceValidationListener());
        } else {
            boolean found = false;
            for(Listener l : listeners){
                if(l instanceof NonInplaceValidationListener){
                    found = true;
                    break;
                }
            }
            if(!found){
                sameDiff.addListeners(new NonInplaceValidationListener());
            }
        }

        //Check forward pass:
        if (testCase.fwdTestFns() != null && testCase.fwdTestFns().size() > 0) {
            SameDiff sd = testCase.sameDiff();

            //Collect variables we need outputs for...
            Set<String> reqVars = testCase.fwdTestFns().keySet();

            Map<String,INDArray> out;
            try {
                out = sd.output(testCase.placeholderValues(), new ArrayList<>(reqVars));
            } catch (Exception e) {
                throw new RuntimeException("Error during forward pass testing" + testCase.testNameErrMsg(), e);
            }

            for (Map.Entry<String, Function<INDArray, String>> e : testCase.fwdTestFns().entrySet()) {
                SDVariable v = sd.getVariable(e.getKey());
                if (v == null) {
                    throw new IllegalStateException("Test case has expected result function defined for variable \"" +
                            e.getKey() + "\" but SameDiff instance does not have a variable for this name" + testCase.testNameErrMsg());
                }

                INDArray actual = out.get(v.name());
                if (actual == null) {
                    throw new IllegalStateException("Null INDArray after forward pass for variable \"" + e.getKey() + "\"");
                }

                String error;
                try {
                    error = e.getValue().apply(actual);
                } catch (Throwable t) {
                    throw new IllegalStateException("Error checking forward pass for variable \"" + e.getKey() + "\": exception was" +
                            " thrown by foward pass validation function", t);
                }

                if (error != null) {
                    return testCase.testNameErrMsg() + ": Variable " + e.getKey() + " failed: " + error;
                }
            }

            ByteBuffer serializedAfterExec = null;
            if(testCase.testFlatBufferSerialization() == TestCase.TestSerialization.BEFORE_EXEC || testCase.testFlatBufferSerialization() == TestCase.TestSerialization.BOTH){
                serializedAfterExec = testCase.sameDiff().asFlatBuffers(true);
                Preconditions.checkNotNull(serializedAfterExec, "Serialization failed? Null output");
            }

            //Now: deserialize, and check the results
            if(serializedBeforeExec != null){
                checkDeserializedEquality(sd, serializedBeforeExec, testCase);
            }
        }

        //Check gradients:
        if (testCase.gradientCheck()) {
            boolean ok;
            try {
                ok = GradCheckUtil.checkGradients(testCase);
            } catch (Throwable t) {
                t.printStackTrace();
                throw new IllegalStateException("Exception encountered during gradient check" + testCase.testNameErrMsg(), t);
            }

            if (!ok) {
                return "Gradient check failed" + testCase.testNameErrMsg();
            }
        }

        return null;    //OK - passed
    }

    public static void checkDeserializedEquality(SameDiff original, ByteBuffer bbSerialized, TestCase tc) {
        SameDiff deserialized;
        try{
            deserialized = SameDiff.fromFlatBuffers(bbSerialized);
        } catch (IOException e){
            throw new RuntimeException("IOException deserializing from FlatBuffers", e);
        }

        //Check variables:
        List<SDVariable> vars = original.variables();
        List<SDVariable> varsDe = deserialized.variables();
        Preconditions.checkState(vars.size() == varsDe.size(), "Number of variables differs: expected %s, got %s", vars.size(), varsDe.size());
        for( int i=0; i<vars.size(); i++ ){
            SDVariable vO = vars.get(i);
            SDVariable vD = varsDe.get(i);
            Preconditions.checkState(vO.name().equals(vD.name()), "Names should be equal for variable %s: expected %s vs %s",
                    i, vO.name(), vD.name());
        }

        //Check ops:
        Map<String,SameDiffOp> opsOrig = original.getOps();
        Map<String,SameDiffOp> opsDeser = deserialized.getOps();
        Preconditions.checkState(opsOrig.keySet().equals(opsDeser.keySet()), "Op names differs: %s vs. %s", opsOrig.keySet(), opsDeser.keySet());

        for(String s : opsOrig.keySet()){
            SameDiffOp orig = opsOrig.get(s);
            SameDiffOp des = opsDeser.get(s);
            Preconditions.checkState(orig.getName().equals(des.getName()), "Names differ: %s vs %s", orig.getName(), des.getName());
            Preconditions.checkState((orig.getInputsToOp() == null) == (des.getInputsToOp() == null), "Inputs differ: %s vs. %s", orig.getInputsToOp(), des.getInputsToOp());
            Preconditions.checkState(orig.getInputsToOp() == null || orig.getInputsToOp().equals(des.getInputsToOp()), "Inputs differ: %s vs. %s", orig.getInputsToOp(), des.getInputsToOp());

            Preconditions.checkState((orig.getOutputsOfOp() == null) == (des.getOutputsOfOp() == null), "Outputs differ: %s vs. %s", orig.getOutputsOfOp(), des.getOutputsOfOp());
            Preconditions.checkState(orig.getOutputsOfOp() == null || orig.getOutputsOfOp().equals(des.getOutputsOfOp()), "Outputs differ: %s vs. %s", orig.getOutputsOfOp(), des.getOutputsOfOp());

            Preconditions.checkState((orig.getControlDeps() == null) == (des.getControlDeps() == null), "Control dependencies differ: %s vs. %s", orig.getControlDeps(), des.getControlDeps());
            Preconditions.checkState(orig.getControlDeps() == null || orig.getControlDeps().equals(des.getControlDeps()), "Control dependencies differ: %s vs. %s", orig.getControlDeps(), des.getControlDeps());

            Preconditions.checkState((orig.getVarControlDeps() == null) == (des.getVarControlDeps() == null), "Op variable control dependencies differ: %s vs. %s", orig.getVarControlDeps(), des.getVarControlDeps());
            Preconditions.checkState(orig.getVarControlDeps() == null || orig.getVarControlDeps().equals(des.getVarControlDeps()), "Op variable control dependencies differ: %s vs. %s", orig.getControlDeps(), des.getControlDeps());

            Preconditions.checkState((orig.getControlDepFor() == null) == (des.getControlDepFor() == null), "Op control dependencies for list differ: %s vs. %s", orig.getControlDepFor(), des.getControlDepFor());
            Preconditions.checkState(orig.getControlDepFor() == null || orig.getControlDepFor().equals(des.getControlDepFor()), "Op variable control dependencies differ: %s vs. %s", orig.getControlDepFor(), des.getControlDepFor());

            Preconditions.checkState(orig.getOp().getClass() == des.getOp().getClass(), "Classes differ: %s v. %s", orig.getOp().getClass(), des.getOp().getClass());
        }

        //Check placeholders:
        Set<String> phBefore = new HashSet<>();
        Set<String> phAfter = new HashSet<>();

        for(Variable v : original.getVariables().values()){
            if(v.getVariable().isPlaceHolder())
                phBefore.add(v.getName());
        }
        for(Variable v : deserialized.getVariables().values()){
            if(v.getVariable().isPlaceHolder())
                phAfter.add(v.getName());
        }

        if(phBefore == null){
            Preconditions.checkState(phAfter == null || phAfter.size() == 0, "%s", phAfter);
        } else {
            Preconditions.checkState(phAfter != null, "Placeholders after deserialization was null");
            Preconditions.checkState(phBefore.equals(phAfter), "Before: %s, after deserialization: %s", phBefore, phAfter);
        }

        Map<String,Variable> varsBefore = original.getVariables();
        Map<String,Variable> varsAfter = deserialized.getVariables();
        Preconditions.checkState(varsBefore.keySet().equals(varsAfter.keySet()), "Variable keysets do not match: %s vs %s", varsBefore.keySet(), varsAfter.keySet());

//        System.out.println(original.summary());
//        System.out.println("\n\n\n\n");
//        System.out.println(deserialized.summary());

        for(String s : varsBefore.keySet()){
            Variable vB = varsBefore.get(s);
            Variable vA = varsAfter.get(s);
            Preconditions.checkState(vB.getName().equals(vA.getName()), "Variable names do not match: %s vs %s", vA.getName(), vB.getName());
            Preconditions.checkState(vB.getVariable().getVariableType() == vA.getVariable().getVariableType(),
                    "Variable types do not match: %s - %s vs %s", s, vB.getVariable().getVariableType(), vA.getVariable().getVariableType());

            equalConsideringNull(vB.getInputsForOp(), vA.getInputsForOp(), "%s - Input to ops differ: %s vs. %s", s, vB.getInputsForOp(), vA.getInputsForOp());

            Preconditions.checkState((vB.getOutputOfOp() == null && vA.getOutputOfOp() == null) || vB.getOutputOfOp().equals(vA.getOutputOfOp()), "%s - Output of op differ: %s vs. %s", s, vB.getOutputOfOp(), vA.getOutputOfOp());

            equalConsideringNull(vB.getControlDeps(), vA.getControlDeps(), "%s - Control dependencies differ: %s vs. %s", s, vB.getControlDeps(), vA.getControlDeps());

            equalConsideringNull(vB.getControlDepsForOp(), vA.getControlDepsForOp(), "%s - Control dependencies for ops differ: %s vs. %s", s, vB.getControlDepsForOp(), vA.getControlDepsForOp());

            equalConsideringNull(vB.getControlDepsForVar(), vA.getControlDepsForVar(), "%s - Control dependencies for vars differ: %s vs. %s", s, vB.getControlDepsForVar(), vA.getControlDepsForVar());
        }

        //Check loss variables:
        List<String> lossVarBefore = original.getLossVariables();
        List<String> lossVarAfter = deserialized.getLossVariables();
        if(lossVarBefore == null || lossVarBefore.isEmpty()){
            Preconditions.checkState(lossVarAfter == null || lossVarAfter.isEmpty(), "Loss variables ");
        } else {
            Preconditions.checkState(lossVarBefore.equals(lossVarAfter), "Loss variables are not equal after deserialization: %s vs %s",
                    lossVarBefore, lossVarAfter);
        }

        if(tc.fwdTestFns() != null && !tc.fwdTestFns().isEmpty()) {
            //Finally: check execution/output
            Map<String,INDArray> outOrig = original.outputAll(tc.placeholderValues());
            Map<String,INDArray> outDe = deserialized.outputAll(tc.placeholderValues());
            Preconditions.checkState(outOrig.keySet().equals(outDe.keySet()), "Keysets for execution after deserialization does not match key set for original model");

            for (String s : outOrig.keySet()) {
                INDArray orig = outOrig.get(s);
                INDArray deser = outDe.get(s);

                Function<INDArray, String> f = tc.fwdTestFns().get(s);
                String err = null;
                if (f != null) {
                    err = f.apply(deser);
                } else {
                    if (!orig.equals(deser)) {
                        //Edge case: check for NaNs in original and deserialized... might be legitimate test (like replaceNaNs op)
                        long count = orig.dataType().isNumerical() ? Nd4j.getExecutioner().execAndReturn(new MatchCondition(orig, Conditions.isNan())).getFinalResult().longValue() : -1;
                        if (orig.dataType().isNumerical() && count > 0 && orig.equalShapes(deser)) {
                            long count2 = Nd4j.getExecutioner().execAndReturn(new MatchCondition(deser, Conditions.isNan())).getFinalResult().longValue();
                            if (count != count2) {
                                err = "INDArray equality failed";
                            } else {
                                //TODO is there a better way to do this?
                                NdIndexIterator iter = new NdIndexIterator(orig.shape());
                                while (iter.hasNext()) {
                                    long[] i = iter.next();
                                    double d1 = orig.getDouble(i);
                                    double d2 = deser.getDouble(i);
                                    if ((Double.isNaN(d1) != Double.isNaN(d2)) || (Double.isInfinite(d1) != Double.isInfinite(d2)) || Math.abs(d1 - d2) > 1e-5) {
                                        err = "INDArray equality failed";
                                        break;
                                    }
                                }
                            }
                        } else {
                            err = "INDArray equality failed";
                        }
                    }
                }

                Preconditions.checkState(err == null, "Variable result (%s) failed check - \"%ndSInfo\" vs \"%ndSInfo\" - %nd10 vs %nd10\nError:%s", s, orig, deser, orig, deser, err);
            }
        }
    }

    protected static void equalConsideringNull(List<String> l1, List<String> l2, String msg, Object... args){
        //Consider null and length 0 list to be equal (semantically they mean the same thing)
        boolean empty1 = l1 == null || l1.isEmpty();
        boolean empty2 = l2 == null || l2.isEmpty();
        if(empty1 && empty2){
            return;
        }
        Preconditions.checkState(l1 == null || l1.equals(l2), msg, args);
    }

    /**
     * Validate the outputs of a single op
     *
     * @param testCase Op test case to run
     * @return NULL if test is OK, or an error message otherwise
     */
    public static String validate(OpTestCase testCase) {
        collectCoverageInformation(testCase);

        //Check shape function:
        List<LongShapeDescriptor> outShapes;
        try {
            outShapes = Nd4j.getExecutioner().calculateOutputShape(testCase.op());
        } catch (Throwable t) {
            throw new IllegalStateException("Error calculating output shapes during op validation", t);
        }

        if (outShapes.size() != testCase.testFns().size()) {
            return "Expected number of output shapes and number of outputs differ. " + outShapes.size() + " output shapes," +
                    " but OpTestCase specifies " + testCase.testFns().size() + " outputs expected";
        }

        for (int i = 0; i < outShapes.size(); i++) {
            val act = outShapes.get(i);
            val exp = testCase.expShapes().get(i);
            if(!Objects.equals(exp.dataType(), act.dataType())){
                return "Shape function check failed for output " + i + ": expected shape " + exp + ", actual shape " + act;
            }
            if(!Arrays.equals(act.getShape(), exp.getShape())){
                return "Shape function check failed for output " + i + ": expected shape " + exp + ", actual shape " + act;
            }
        }

        //Check the outputs:
        try {
            Nd4j.getExecutioner().execAndReturn(testCase.op());
        } catch (Throwable t) {
            throw new IllegalStateException("Error during op execution", t);
        }

        for (int i = 0; i < testCase.testFns().size(); i++) {
            String error;
            try {
                error = testCase.testFns().get(i).apply(testCase.op().outputArguments().get(i));
            } catch (Throwable t) {
                throw new IllegalStateException("Exception thrown during op output validation for output " + i, t);
            }

            if (error != null) {
                return "Output " + i + " failed: " + error;
            }
        }

        return null;    //OK
    }


    //==================================================================================================================
    // Coverage information

    private static List<Class> allOps;
    private static List<Long> nonMappedLibnd4jOps;
    private static Map<Long,Pair<List<String>,CustomOpDescriptor>> dedupedCustomOps;
    private static int countTotalLibnd4jOps;
    private static Map<Class, Integer> gradCheckCoverageCountPerClass = new LinkedHashMap<>();
    private static Map<Class, Integer> fwdPassCoverageCountPerClass = new LinkedHashMap<>();
    private static Map<Class, Integer> singleOpTestCountPerClass = new LinkedHashMap<>();
    private static Map<Class, Integer> opsWithTFMappingTFImportCounts = new LinkedHashMap<>();
    private static Map<String, Integer> tfMappedOpsImportTestCounts = new LinkedHashMap<>();


    private static void collectCoverageInformation(TestCase testCase) {
        SameDiff sd = testCase.sameDiff();

        //NOTE: Count on a per-test-case basis, not on a 'per function seen' basis
        //i.e., don't double count if a SameDiff instance has multiple copies of the same op type

        //Collect coverage information for backprop:
        DifferentialFunction[] functions = sd.ops();
        Set<Class> backpropSeen = new HashSet<>();
        for (DifferentialFunction df : functions) {
            backpropSeen.add(df.getClass());
        }
        for (Class c : backpropSeen) {
            if(gradCheckCoverageCountPerClass.containsKey(c))
                gradCheckCoverageCountPerClass.put(c, gradCheckCoverageCountPerClass.get(c) + 1);
            else
                gradCheckCoverageCountPerClass.put(c, 1);
        }

        //Collect coverage information for forward pass (expected outputs)
        Set<Class> seen = null;
        if (testCase.fwdTestFns() != null) {
            for (String s : testCase.fwdTestFns().keySet()) {
                //Determine the differential function that this variable is the output of, if any
                DifferentialFunction df = sd.getVariableOutputOp(s);
                if (df != null) {
                    if (seen == null)
                        seen = new HashSet<>();

                    seen.add(df.getClass());
                }
            }
        }

        if (seen != null) {
            for (Class c : seen) {
                if(fwdPassCoverageCountPerClass.containsKey(c)) {
                    fwdPassCoverageCountPerClass.put(c, fwdPassCoverageCountPerClass.get(c) + 1);
                } else {
                    fwdPassCoverageCountPerClass.put(c, 1);
                }
            }
        }
    }

    private static void collectCoverageInformation(OpTestCase testCase) {
        //TODO we're basically assuming subtypes of DynamicCustomOp here, for coverage... not DCO itself
        if(singleOpTestCountPerClass.containsKey(testCase.op().getClass())) {
            singleOpTestCountPerClass.put(testCase.op().getClass(),
                    singleOpTestCountPerClass.get(testCase.op().getClass()) + 1);
        } else {
            singleOpTestCountPerClass.put(testCase.op().getClass(), 1);
        }
    }


    public static void collectTensorflowImportCoverage(SameDiff graph){
        for(SameDiffOp op : graph.getOps().values()){
            DifferentialFunction d = op.getOp();
            String[] tfNames = null;
            try{
                tfNames = d.tensorflowNames();
            } catch (Throwable t){
                //Ignore
                continue;
            }

            if(tfNames != null && tfNames.length > 0){
                Integer currCount = opsWithTFMappingTFImportCounts.get(d.getClass());
                if(currCount == null)
                    currCount = 0;
                currCount++;
                opsWithTFMappingTFImportCounts.put(d.getClass(), currCount);

                currCount = fwdPassCoverageCountPerClass.get(d.getClass());
                if(currCount == null)
                    currCount = 0;
                currCount++;
                fwdPassCoverageCountPerClass.put(d.getClass(), currCount);

                for(String s : tfNames){
                    currCount = tfMappedOpsImportTestCounts.get(s);
                    if(currCount == null)
                        currCount = 0;
                    currCount++;
                    tfMappedOpsImportTestCounts.put(s, currCount);
                }
            }
        }

    }

    //Collect coverage information
    static {
        initializeCoverage();
    }

    private static void initializeCoverage() {
        //Scan classpath to find all DifferentialFunction instances, so tensorflow/onnx mappings can be made
        //We're assuming here that all instances with such mappings are defined in ND4J
        //As of 04/2018 all DifferentialFunction classes are defined in org.nd4j.linalg.api.ops - with the exception
        // of ILossFunction instances, which don't have TF/Onnx import working anyway
        ImmutableSet<ClassPath.ClassInfo> info;
        try {
            //Dependency note: this ClassPath class was added in Guava 14
            info = org.nd4j.shade.guava.reflect.ClassPath.from(DifferentialFunctionClassHolder.class.getClassLoader())
                    .getTopLevelClassesRecursive("org.nd4j.linalg.api.ops");
        } catch (IOException e) {
            //Should never happen
            throw new RuntimeException(e);
        }

        //Also, info for libnd4j op mapping:
        Map<String,CustomOpDescriptor> customOps = Nd4j.getExecutioner().getCustomOperations();

        //De-duplicate custom ops based on hash (due to aliases also being returned)
        dedupedCustomOps = new HashMap<>();
        for(Map.Entry<String,CustomOpDescriptor> e : customOps.entrySet()){
            long hash = e.getValue().getHash();
            if(!dedupedCustomOps.containsKey(hash)){
                Pair<List<String>,CustomOpDescriptor> p = new Pair<List<String>,CustomOpDescriptor>(new ArrayList<String>(), e.getValue());
                dedupedCustomOps.put(hash, p);
            }
            Pair<List<String>,CustomOpDescriptor> p = dedupedCustomOps.get(hash);
            List<String> l = p.getFirst();
            if(!l.contains(e.getKey())){
                l.add(e.getKey());
            }
        }

        Set<Long> notSeenCustomOps = new HashSet<>(dedupedCustomOps.keySet());

        allOps = new ArrayList<>(gradCheckCoverageCountPerClass.keySet());
        for (ClassPath.ClassInfo c : info) {
            //Load method: Loads (but doesn't link or initialize) the class.
            Class<?> clazz;
            try {
                clazz = Class.forName(c.getName());
            } catch (ClassNotFoundException e) {
                //Should never happen as  this was found on the classpath
                throw new RuntimeException(e);
            }


            if (Modifier.isAbstract(clazz.getModifiers()) || clazz.isInterface() || !DifferentialFunction.class.isAssignableFrom(clazz))
                continue;

            if (DifferentialFunction.class.isAssignableFrom(clazz) && !clazz.getSimpleName().contains("Old")) {   //Exclude OldSubOp, etc
                allOps.add(clazz);
            }

            String opName = null;
            try{
                opName = ((DifferentialFunction)clazz.newInstance()).opName();
            } catch (Exception e){
                log.warn("Could not instantiate object of type {}", clazz.getName(), e);
            }

            if(opName != null){
                CustomOpDescriptor d = customOps.get(opName);
                if(d != null) {
                    notSeenCustomOps.remove(d.getHash());
                }
            }
        }

        countTotalLibnd4jOps = dedupedCustomOps.size();
        nonMappedLibnd4jOps = new ArrayList<>(notSeenCustomOps);
        Collections.sort(nonMappedLibnd4jOps, new Comparator<Long>() {
            @Override
            public int compare(Long o1, Long o2) {
                Pair<List<String>,CustomOpDescriptor> p1 = dedupedCustomOps.get(o1);
                Pair<List<String>,CustomOpDescriptor> p2 = dedupedCustomOps.get(o2);
                return p1.getKey().get(0).compareTo(p2.getKey().get(0));
            }
        });

        Collections.sort(allOps, new Comparator<Class>() {
            @Override
            public int compare(Class o1, Class o2) {
                return o1.getName().compareTo(o2.getName());
            }
        });
        for (Class c : allOps) {
            gradCheckCoverageCountPerClass.put(c, 0);
            fwdPassCoverageCountPerClass.put(c, 0);
            singleOpTestCountPerClass.put(c, 0);
        }
    }

    /**
     * Log the coverage information
     *
     * @param logAdequatelyTested If true: log details of each op that has both forward and (if appropriate) backward tests
     * @param logInadequate       If false: log details of each op that does NOT have both forward and (if appropriate) backward tests
     */
    public static void logCoverageInformation(boolean logAdequatelyTested, boolean logInadequate, boolean logUnmappedLibnd4jOps,
                                              boolean logUntestedTFImport, boolean logUnmappedTFOps) {
        //Set of ops that we can't gradient check
        Set<Class> excludedFromBackpropCoverage = excludedFromGradientCheckCoverage();
        Set<Class> excludedFromAllTestCoverage = excludedFromAllTests();

        String numFormat = "%3d";
        int countAdequate = 0;
        int countAdequateBwd = 0;
        int countAdequateFwd = 0;
        if (logAdequatelyTested) {
            log.info(" --- Adequately Tested Classes ---");
            for (Class c : allOps) {
                if(excludedFromAllTestCoverage.contains(c))
                    continue;

                int countBackpropSeen = gradCheckCoverageCountPerClass.get(c);
                int countFwdValidation = fwdPassCoverageCountPerClass.get(c) + singleOpTestCountPerClass.get(c);

                if (countBackpropSeen > 0) {
                    countAdequateBwd++;
                }
                if (countFwdValidation > 0) {
                    countAdequateFwd++;
                }
                if (countFwdValidation > 0 && countBackpropSeen > 0) {
                    countAdequate++;
                }

                boolean gradExcluded = excludedFromBackpropCoverage.contains(c);
                if (countFwdValidation > 0 && (countBackpropSeen > 0 || gradExcluded)) {
                    //At least 1 forward test, and 1 gradient check

                    if (gradExcluded) {
                        log.info("Forward: {} tests, GradCheck: <excluded> for op {}", String.format(numFormat, countFwdValidation), c.getName());
                    } else {
                        log.info("Forward: {} tests, GradCheck: {} tests  for op {}", String.format(numFormat, countFwdValidation),
                                String.format(numFormat, countBackpropSeen), c.getName());
                    }
                }
            }
        }

        if (logInadequate) {
            log.info(" --- Classes NOT Tested Adequately ---");
            for (Class c : allOps) {
                if(excludedFromAllTestCoverage.contains(c))
                    continue;
                int countBackpropSeen = gradCheckCoverageCountPerClass.get(c);
                int countFwdValidation = fwdPassCoverageCountPerClass.get(c) + singleOpTestCountPerClass.get(c);

                boolean gradExcluded = excludedFromBackpropCoverage.contains(c);
                if (countFwdValidation == 0 || (countBackpropSeen == 0 && !gradExcluded)) {
                    //0 forward test OR 0 gradient check (and not excluded from grad checks)

                    if (gradExcluded) {
                        log.info("Forward: {} tests, GradCheck: <excluded> for op {}", String.format(numFormat, countFwdValidation), c.getName());
                    } else {
                        log.info("Forward: {} tests, GradCheck: {} tests  for op {}", String.format(numFormat, countFwdValidation),
                                String.format(numFormat, countBackpropSeen), c.getName());
                    }
                }
            }
        }

        int countLibnd4jIgnored = 0;
        if(logUnmappedLibnd4jOps ){
            Set<String> ignoreLibnd4j = excludeFromLibnd4jCustomOpMapping();
            log.info(" --- Libnd4j Ops Not Mapped ---");
            for(long l : nonMappedLibnd4jOps){
                Pair<List<String>,CustomOpDescriptor> p = dedupedCustomOps.get(l);
                boolean foundIgnore = false;
                for(String s : p.getFirst()){
                    if(ignoreLibnd4j.contains(s)){
                        foundIgnore = true;
                        countLibnd4jIgnored++;
                        break;
                    }
                }
                if(foundIgnore)
                    continue;
                log.info("Not mapped libnd4j custom op: {} (hash: {})", p.getFirst(), l);
            }
        }

        //Log info for TF import op coverage:
        Map<String,DifferentialFunction> tfOpsMap = DifferentialFunctionClassHolder.getInstance().getTensorFlowNames();
        int totalTFMappedOps = tfOpsMap.size();
        int tfOpsWithImportTests = 0;
        if(logUntestedTFImport)
            log.info(" --- Ops with TF Mapping but No TF Import Tests ---");
        List<String> tfOpsKeys = new ArrayList<>(tfOpsMap.keySet());
        Collections.sort(tfOpsKeys);
        Set<String> tfIgnored = excludeFromTfImportCoverage();
        int tfImportIgnored = 0;
        for(String s : tfOpsKeys){
            Integer count = tfMappedOpsImportTestCounts.get(s);
            if(count == null || count == 0){
                if(tfIgnored.contains(s)){
                    tfImportIgnored++;
                } else if(logUntestedTFImport)
                    log.info("TF mapped op with no import tests: {}", s);
            } else {
                tfOpsWithImportTests++;
            }
        }

        if(logUnmappedTFOps){
            log.info(" --- TF Ops Not Mapped for Import ---");
            Map<String,OpDef> allTFOps;
            try{
                allTFOps = TensorflowDescriptorParser.opDescs();
            } catch (Throwable t){
                throw new RuntimeException(t);
            }

            List<String> notMapped = new ArrayList<>();
            for(String s : allTFOps.keySet()){
                if(DifferentialFunctionClassHolder.getInstance().getOpWithTensorflowName(s) == null &&
                        !tfIgnored.contains(s)){
                    notMapped.add(s);
                }
            }

            Collections.sort(notMapped);
            int subsets = (int)Math.ceil(notMapped.size() / 10);
            for( int i=0; i<subsets; i++ ){
                log.info("TF ops not mapped for import: {}", notMapped.subList(10*i, Math.min(10*(i+1), notMapped.size())));
            }
        }


        int totalFwd = 0;
        for(Class c : allOps){
            if(!excludedFromAllTestCoverage.contains(c))
                totalFwd++;
        }
        int totalBwd = 0;
        for (Class c : allOps) {
            if (!isBackpropOp(c)) {
                totalBwd++;
            }
        }

        double fracFwdAdequate = countAdequateFwd / (double) totalFwd;
        double fracBwdAdequate = countAdequateBwd / (double) totalBwd;
        double fracAdequate = countAdequate / (double) allOps.size();
        String pcFwd = String.format("%.2f", fracFwdAdequate * 100.0);
        String pcBwd = String.format("%.2f", fracBwdAdequate * 100.0);
        String pc = String.format("%.2f", fracAdequate * 100.0);

        int countTf = DifferentialFunctionClassHolder.getInstance().getCountTotalTfOps();
        int countTfMapped = DifferentialFunctionClassHolder.getInstance().getCountTotalMappedOps();
        double tfFrac = countTfMapped / (double)countTf;
        String fracTfStr = String.format("%.2f", 100.0 * tfFrac);

        int countLibnd4jMapped = countTotalLibnd4jOps - nonMappedLibnd4jOps.size();
        String fracLibnd4j = String.format("%.2f", 100.0 * (countLibnd4jMapped / (double)(countTotalLibnd4jOps - countLibnd4jIgnored)));

        String fracTFMappedTested = String.format("%.2f", 100.0 * tfOpsWithImportTests / (double)(totalTFMappedOps-tfImportIgnored));

        log.info("*****************************************************");
        log.info("Op Validation:                        {} of {} classes with adequate tests ({}% coverage)", countAdequate, totalFwd, pc);
        log.info("Forward pass tests:                   {} of {} classes ({}% coverage)", countAdequateFwd, totalFwd, pcFwd);
        log.info("Gradient check tests:                 {} of {} classes ({}% coverage)", countAdequateBwd, totalBwd, pcBwd);
        log.info("({} ops excluded from gradient check coverage)", excludedFromBackpropCoverage.size());
        log.info("({} ops excluded from fwd+gradient tests)", excludedFromAllTestCoverage.size());
        log.info("TF mapped ops:                        {} of {} ({}%)", countTfMapped, countTf, fracTfStr);
        log.info("SD ops with TF import mapping + test  {} of {} ({}%) - {} ignored for coverage", tfOpsWithImportTests, (totalTFMappedOps-tfImportIgnored), fracTFMappedTested, tfImportIgnored);
        log.info("Libnd4j mapped ops:                   {} of {} ({}%) - {} excluded for coverage", countLibnd4jMapped, countTotalLibnd4jOps, fracLibnd4j, countLibnd4jIgnored);
        log.info("*****************************************************");
    }

    private static boolean isBackpropOp(Class<?> c) {
        String name = c.getSimpleName();
        return name.contains("Bp") || name.contains("Derivative") || name.contains("Grad");
    }


    private static Set<Class> excludedFromAllTests() {
        List list = Arrays.asList(
                //Exclude misc
                DynamicCustomOp.class,
                GradientBackwardsMarker.class,
                EqualsWithEps.class,
                FreeGridOp.class,
                MergeSum.class, //Redundant; we use MergeAdd in samediff instead
                ScalarRemainder.class,  //Redundant; SameDiff uses ScalarFMod instead
                RestoreV2.class,
                SaveV2.class,
                ScalarSetValue.class,   //Not used in SameDiff (it's a "set to X if less than X" type op, redundant given other ops)
                BinomialDistributionEx.class,   //Redundant?

                //Exclude manual broadcast ops: SameDiff uses auto broadcasting
                BroadcastAMax.class,
                BroadcastAMin.class,
                BroadcastAddOp.class,
                BroadcastCopyOp.class,
                BroadcastDivOp.class,
                BroadcastEqualTo.class,
                BroadcastGreaterThan.class,
                BroadcastGreaterThanOrEqual.class,
                BroadcastLessThan.class,
                BroadcastLessThanOrEqual.class,
                BroadcastMax.class,
                BroadcastMin.class,
                BroadcastMulOp.class,
                BroadcastNotEqual.class,
                BroadcastRDivOp.class,
                BroadcastRSubOp.class,
                BroadcastSubOp.class,

                //These BP ops: we'll test them as part of gradient checks for the corresponding forward pass ops
                //We don't need separate forward pass tests (as long as  gradient checks pass), and can't gradient check
                // them separately to the forward ops anyway
                AddBpOp.class,
                DivBpOp.class,
                FloorDivBpOp.class,
                FloorModBpOp.class,
                MulBpOp.class,
                RDivBpOp.class,
                RSubBpOp.class,
                SquaredDifferenceBpOp.class,
                SubBpOp.class,
                CumProdBp.class,
                DotBp.class,
                SquaredNormBp.class,
                SoftmaxBp.class,

                CubeDerivative.class,
                GELUDerivative.class,
                PreciseGELUDerivative.class,
                HardSigmoidDerivative.class,
                HardTanhDerivative.class,
                LeakyReLUDerivative.class,
                LogSoftMaxDerivative.class,
                RationalTanhDerivative.class,
                RectifiedTanhDerivative.class,
                Relu6Derivative.class,
                PReluBp.class,
                SELUDerivative.class,
                SigmoidDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.SigmoidDerivative.class,
                SoftSignDerivative.class,
                TanhDerivative.class,
                SwishDerivative.class,
                TanDerivative.class,
                TanhDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.strict.TanhDerivative.class,
                PowDerivative.class,
                org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinearDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.CubeBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.EluBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.HardSigmoidBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.HardTanhBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.LeakyReLUBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.RationalTanhBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.RectifiedTanhBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.SeluBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftPlusBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftSignBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.gradient.ThresholdReluBp.class,
                org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.ModBpOp.class,


                BiasAddGrad.class,
                ConcatBp.class,
                TileBp.class,

                BatchNormDerivative.class,
                Conv2DDerivative.class,
                Conv3DDerivative.class,
                DeConv2DDerivative.class,
                LocalResponseNormalizationDerivative.class,
                Pooling2DDerivative.class,
                Pooling3DDerivative.class,
                SConv2DDerivative.class,
                Upsampling2dDerivative.class,
                Im2colBp.class,

                SliceBp.class,
                StridedSliceBp.class,
                MmulBp.class,
                DotProductAttentionBp.class,
                MultiHeadDotProductAttentionBp.class,
                LayerNormBp.class,
                StandardizeBp.class,
                DynamicPartitionBp.class,

                AbsoluteDifferenceLossBp.class,
                CosineDistanceLossBp.class,
                HingeLossBp.class,
                HuberLossBp.class,
                LogLossBp.class,
                LogPoissonLossBp.class,
                MeanPairwiseSquaredErrorLossBp.class,
                MeanSquaredErrorLossBp.class,
                SigmoidCrossEntropyLossBp.class,
                SoftmaxCrossEntropyLossBp.class,
                SparseSoftmaxCrossEntropyLossWithLogitsBp.class,

                SegmentMaxBp.class,
                SegmentMeanBp.class,
                SegmentMinBp.class,
                SegmentProdBp.class,
                SegmentSumBp.class,
                UnsortedSegmentMaxBp.class,
                UnsortedSegmentMeanBp.class,
                UnsortedSegmentMinBp.class,
                UnsortedSegmentProdBp.class,
                UnsortedSegmentSqrtNBp.class,
                UnsortedSegmentSumBp.class,

                //Not intended for general users; only used in DL4J SameDiff integration + tested adequately there
                ExternalErrorsFunction.class,

                //Meta-Ops: not available in SameDiff
                InvertedPredicateMetaOp.class,
                PostulateMetaOp.class,
                PredicateMetaOp.class,
                ReduceMetaOp.class,


                //Ops not intended to be used in SameDiff:
                BarnesEdgeForces.class,
                BarnesHutGains.class,
                BarnesHutSymmetrize.class,
                SpTreeCell.class,
                CbowRound.class,
                SkipGramRound.class,
                HashCode.class
        );

        return new HashSet<>(list);
    }

    /**
     * Returns a list of classes that are not gradient checkable.
     * An operation may not be gradient checkable due to, for example:
     * (a) Having no real-valued arguments<br>
     * (b) Having random output (dropout, for example)<br>
     * <p>
     * Note that hawving non-real-valued output is OK - we still want to test these, as they
     * should pass back zero gradients!
     */
    private static Set<Class> excludedFromGradientCheckCoverage() {
        List list = Arrays.asList(
                //Exclude misc
                DynamicCustomOp.class,
                EqualsWithEps.class,
                ConfusionMatrix.class,
                Eye.class,
                OneHot.class,
                BinaryMinimalRelativeError.class,
                BinaryMinimalRelativeError.class,
                InvertPermutation.class,    //Uses integer indices
                ConfusionMatrix.class,      //Integer indices
                Linspace.class,             //No input array
                Assert.class,
                //Exclude boolean operations, boolean reductions, etc:
                Any.class,
                All.class,
                IsInf.class,
                org.nd4j.linalg.api.ops.impl.transforms.bool.IsInf.class,
                IsNaN.class,
                org.nd4j.linalg.api.ops.impl.transforms.bool.IsNaN.class,
                BooleanNot.class,
                Not.class,
                MatchConditionTransform.class,
                InTopK.class,
                IsNonDecreasing.class,
                IsStrictlyIncreasing.class,
                IsNumericTensor.class,
                //Exclude index accumulations (index out, not real-valued)
                FirstIndex.class,
                IAMax.class,
                IAMin.class,
                IMax.class,
                IMin.class,
                LastIndex.class,
                ArgMax.class,
                ArgMin.class,

                //Exclude ops that output integer types only:
                Shape.class,
                ShapeN.class,
                SizeAt.class,
                BroadcastDynamicShape.class,
                ReductionShape.class,
                ShiftBits.class,
                RShiftBits.class,
                BitsHammingDistance.class,
                CyclicShiftBits.class,
                CyclicRShiftBits.class,


                //Exclude Random ops
                RandomStandardNormal.class,
                DistributionUniform.class,
                AlphaDropOut.class,
                BernoulliDistribution.class,
                BinomialDistribution.class,
                BinomialDistributionEx.class,
                Choice.class,
                DropOut.class,
                DropOutInverted.class,
                GaussianDistribution.class,
                LogNormalDistribution.class,
                ProbablisticMerge.class,
                Range.class,
                TruncatedNormalDistribution.class,
                UniformDistribution.class,
                //Other ops we don't intend to be differentiable (only used as part of backprop, etc).
                // But we still want a forward/check for these
                Col2Im.class,
                NormalizeMoments.class,  //In principle differentiable. In practice: doesn't make any sense to do so!
                CumProdBp.class,
                CumSumBp.class,
                DotBp.class,
                MaxBp.class,
                MeanBp.class,
                MinBp.class,
                Norm1Bp.class,
                Norm2Bp.class,
                NormMaxBp.class,
                ProdBp.class,
                StandardDeviationBp.class,
                SumBp.class,
                VarianceBp.class,

                LogicalAnd.class,
                LogicalNot.class,
                LogicalOr.class,
                LogicalXor.class,

                Histogram.class
        );

        return new HashSet<>(list);
    }

    /**
     * These ops are excluded from TF import test coverage, for various reasons
     */
    private static Set<String> excludeFromTfImportCoverage(){
        List<String> list = Arrays.asList(
                "Reverse",      //Can be excluded because "Reverse_v2" is synonym that TF uses with tf.reverse(...); ReverseV2 is also Java op that is synonym for same op
                "LogSigmoid",    //Not in ops.proto. Have tests for tf.log_sigmoid, but can't test LogSigmoid op directly: tf.log_sigmoid actually just uses "y = -tf.nn.softplus(-x)" - i.e., 3 separate ops :/
                "HardSigmoid",   //Also implemented as python, NOT a single native op
                "SpaceToBatch", //Old name - SpaceToBatchNd is used in practice (inc. for tf.space_to_batch)
                "BatchToSpace", //Old name - BatchToSpaceNd is used in practice
                "Pad",          //As far as I can tell: Only PadV2 and MirrorPad are used in practice
                "TopK",         //TopKV2 used
                "InTopK",       //InTopKV2 used
                "BatchMatrixDeterminant",   //Deprecated in favor of "MatrixDeterminant"
                "BatchMatrixDiagPart",      //Deprecated in favor of "MatrixDiagPart"
                "BatchMatrixDiag",          //Deprecated in favor of "MatrixDiag"
                "BatchMatrixBandPart",      //Deprecated in favor of "MatrixBandPart"
                "BatchMatrixInverse",       //Deprecated in favor of "MatrixInverse"
                "BatchMatrixSetDiag",       //Deprecated in favor of "MatrixSetDiag"
                "BatchMatrixSolve",         //Deprecated in favor of "MatrixSolve"
                "BatchMatrixSolveLs",       //Deprecated in favor of "MatrixSolveLs"
                "BatchMatrixTriangularSolve",   //Deprecated in favor of "MatrixTriangularSolve"
                "BatchSelfAdjointEig",      //Deprecated in favor of "SelfAdjointEigV2"
                "BatchSelfAdjointEigV2",    //Deprecated in favor of "SelfAdjointEigV2"
                "BatchSvd",                 //Deprecated in favor of "Svd"

                //These we will likely neven support importing
                "ExperimentalBytesProducedStatsDataset",
                "ExperimentalCSVDataset",
                "ExperimentalDatasetCardinality",
                "ExperimentalDatasetToTFRecord",
                "ExperimentalDenseToSparseBatchDataset",
                "ExperimentalDirectedInterleaveDataset",
                "ExperimentalGroupByReducerDataset",
                "ExperimentalGroupByWindowDataset",
                "ExperimentalIdentityIndexedDataset",
                "ExperimentalIgnoreErrorsDataset",
                "ExperimentalIndexedDatasetGet",
                "ExperimentalIndexedDatasetMaterialize",
                "ExperimentalIteratorGetDevice",
                "ExperimentalLMDBDataset",
                "ExperimentalLatencyStatsDataset",
                "ExperimentalMapAndBatchDataset",
                "ExperimentalMapDataset",
                "ExperimentalMatchingFilesDataset",
                "ExperimentalMaterializedIndexDatasetHandle",
                "ExperimentalMaxIntraOpParallelismDataset",
                "ExperimentalNonSerializableDataset",
                "ExperimentalNumaMapAndBatchDataset",
                "ExperimentalParallelInterleaveDataset",
                "ExperimentalParseExampleDataset",
                "ExperimentalPrivateThreadPoolDataset",
                "ExperimentalRandomDataset",
                "ExperimentalScanDataset",
                "ExperimentalSetStatsAggregatorDataset",
                "ExperimentalSleepDataset",
                "ExperimentalSlidingWindowDataset",
                "ExperimentalSqlDataset",
                "ExperimentalStatsAggregatorHandle",
                "ExperimentalStatsAggregatorSummary",
                "ExperimentalThreadPoolDataset",
                "ExperimentalThreadPoolHandle",
                "ExperimentalUnbatchDataset",
                "ExperimentalUniqueDataset",

                "DebugIdentity",
                "NcclAllReduce",
                "NcclBroadcast",
                "NcclReduce",

                //Can't import these without embedding entire python runtime and dependencies
                "PyFunc",
                "PyFuncStateless",

                //"QuantizedX" ops are deprecated / no longer supported ("standard" ops have quantized support in many cases)
                "QuantizedAdd",
                "QuantizedAvgPool",
                "QuantizedBatchNormWithGlobalNormalization",
                "QuantizedBiasAdd",
                "QuantizedConcat",
                "QuantizedConv2D",
                "QuantizedInstanceNorm",
                "QuantizedMatMul",
                "QuantizedMaxPool",
                "QuantizedMul",
                "QuantizedRelu",
                "QuantizedRelu6",
                "QuantizedReluX",
                "QuantizedReshape",
                "QuantizedResizeBilinear",


                //All of the following ops - not available in TF (can't find them) - op mapping is wrong?
                //TODO: Check these and remove the import mapping from the Java classes if they are indeed bad
                "HardTanh",
                "Swish",
                "RDiv",
                "DivScalar",
                "LogX",
                "RationalTanh",
                "absargmax",
                "absargmin",
                "entropy_shannon",   //This is a thing, but quite different from our op: https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/contrib/bayesflow/entropy/entropy_shannon
                "count_zero"
        );

        return new HashSet<>(list);
    }


    /**
     * These ops are ones we will never map at Java level for one reason or another
     */
    private static Set<String> excludeFromLibnd4jCustomOpMapping(){
        Set<String> out = new HashSet<>();
        Collections.addAll(out,
                //Test and misc ops:
                "TestOp2i2o", "testop2i2o",
                "firas_sparse",
                "test_output_reshape",
                "test_scalar",
                "testcustom",
                "testreduction",

                //"to_x" ops - we'll use cast instead in SameDiff (which supports all dtypes)
                "to_double",
                "to_float16",
                "to_float32",
                "to_int32",
                "to_int64",
                "to_uint32",
                "to_uint64"
        );

        return out;
    }

}
