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

import com.google.common.collect.ImmutableSet;
import com.google.common.reflect.ClassPath;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.functions.DifferentialFunctionFactory;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOpDescriptor;
import org.nd4j.linalg.api.ops.DefaultOpConverter;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.accum.All;
import org.nd4j.linalg.api.ops.impl.accum.Any;
import org.nd4j.linalg.api.ops.impl.accum.EqualsWithEps;
import org.nd4j.linalg.api.ops.impl.accum.NormalizeMoments;
import org.nd4j.linalg.api.ops.impl.accum.bp.*;
import org.nd4j.linalg.api.ops.impl.broadcast.*;
import org.nd4j.linalg.api.ops.impl.grid.FreeGridOp;
import org.nd4j.linalg.api.ops.impl.indexaccum.*;
import org.nd4j.linalg.api.ops.impl.layers.convolution.*;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarRemainder;
import org.nd4j.linalg.api.ops.impl.scalar.comparison.ScalarSetValue;
import org.nd4j.linalg.api.ops.impl.shape.ConfusionMatrix;
import org.nd4j.linalg.api.ops.impl.shape.Eye;
import org.nd4j.linalg.api.ops.impl.shape.MergeSum;
import org.nd4j.linalg.api.ops.impl.shape.OneHot;
import org.nd4j.linalg.api.ops.impl.shape.bp.ConcatBp;
import org.nd4j.linalg.api.ops.impl.shape.bp.SliceBp;
import org.nd4j.linalg.api.ops.impl.shape.bp.StridedSliceBp;
import org.nd4j.linalg.api.ops.impl.shape.bp.TileBp;
import org.nd4j.linalg.api.ops.impl.transforms.*;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.bp.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.*;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftMaxDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.TanhDerivative;
import org.nd4j.linalg.api.ops.persistence.RestoreV2;
import org.nd4j.linalg.api.ops.persistence.SaveV2;
import org.nd4j.linalg.api.ops.random.compat.RandomStandardNormal;
import org.nd4j.linalg.api.ops.random.custom.DistributionUniform;
import org.nd4j.linalg.api.ops.random.impl.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;
import java.lang.reflect.Modifier;
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


        //Check forward pass:
        if (testCase.fwdTestFns() != null && testCase.fwdTestFns().size() > 0) {
            SameDiff sd = testCase.sameDiff();
            try {
                sd.exec();
            } catch (Exception e) {
                throw new RuntimeException("Error during forward pass testing" + testCase.testNameErrMsg(), e);
            }

            for (Map.Entry<String, Function<INDArray, String>> e : testCase.fwdTestFns().entrySet()) {
                SDVariable v = sd.getVariable(e.getKey());
                if (v == null) {
                    throw new IllegalStateException("Test case has expected result function defined for variable \"" +
                            e.getKey() + "\" but SameDiff instance does not have a variable for this name" + testCase.testNameErrMsg());
                }

                INDArray actual = v.getArr();
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
        }

        //Check gradients:
        if (testCase.gradientCheck()) {
            boolean ok;
            try {
                ok = GradCheckUtil.checkGradients(testCase);
            } catch (Throwable t) {
                throw new IllegalStateException("Exception encountered during gradient check" + testCase.testNameErrMsg(), t);
            }

            if (!ok) {
                return "Gradient check failed" + testCase.testNameErrMsg();
            }
        }

        return null;    //OK - passed
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
        List<long[]> outShapes;
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
            long[] act = outShapes.get(i);
            long[] exp = testCase.expShapes().get(i);
            if (!Arrays.equals(exp, act)) {
                return "Shape function check failed for output " + i + ": expected shape " + Arrays.toString(exp) +
                        ", actual shape " + Arrays.toString(act);
            }
        }

        //Check the outputs:
        try {
            Nd4j.getExecutioner().exec(testCase.op());
        } catch (Throwable t) {
            throw new IllegalStateException("Error during op execution", t);
        }

        for (int i = 0; i < testCase.testFns().size(); i++) {
            String error;
            try {
                error = testCase.testFns().get(i).apply(testCase.op().outputArguments()[i]);
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
        DifferentialFunction[] functions = sd.functions();
        Set<Class> backpropSeen = new HashSet<>();
        for (DifferentialFunction df : functions) {
            backpropSeen.add(df.getClass());
        }
        for (Class c : backpropSeen) {
            gradCheckCoverageCountPerClass.put(c, gradCheckCoverageCountPerClass.get(c) + 1);
        }

        //Collect coverage information for forward pass (expected outputs)
        Set<Class> seen = null;
        if (testCase.fwdTestFns() != null) {
            for (String s : testCase.fwdTestFns().keySet()) {
                //Determine the differential function that this variable is the output of, if any
                DifferentialFunction df = sd.getVariableOutputFunction(s);
                if (df != null) {
                    if (seen == null)
                        seen = new HashSet<>();

                    seen.add(df.getClass());
                }
            }
        }

        if (seen != null) {
            for (Class c : seen) {
                fwdPassCoverageCountPerClass.put(c, fwdPassCoverageCountPerClass.get(c) + 1);
            }
        }
    }

    private static void collectCoverageInformation(OpTestCase testCase) {
        //TODO we're basically assuming subtypes of DynamicCustomOp here, for coverage... not DCO itself
        singleOpTestCountPerClass.put(testCase.op().getClass(),
                singleOpTestCountPerClass.get(testCase.op().getClass()) + 1);
    }


    public static void collectTensorflowImportCoverage(SameDiff graph){

        Map<String,DifferentialFunction> map = graph.getFunctionInstancesById();
        for(DifferentialFunction d : map.values()){
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
            info = com.google.common.reflect.ClassPath.from(DifferentialFunctionClassHolder.class.getClassLoader())
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
    public static void logCoverageInformation(boolean logAdequatelyTested, boolean logInadequate, boolean logUnmappedLibnd4jOps) {
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

        if(logUnmappedLibnd4jOps ){
            log.info(" --- Libnd4j Ops Not Mapped ---");
            for(long l : nonMappedLibnd4jOps){
                Pair<List<String>,CustomOpDescriptor> p = dedupedCustomOps.get(l);
                log.info("Not mapped libnd4j custom op: {} (hash: {})", p.getFirst(), l);
            }
        }

        //Log info for TF import op coverage:
        int totalTFMappedOps = DifferentialFunctionClassHolder.getInstance().getTensorFlowNames().size();
        int tfOpsWithImportTests = 0;
        if(true){       //TODO add config option
            log.info(" --- Ops with TF Mapping but No TF Import Tests ---");
            for(Map.Entry<String, DifferentialFunction> e : DifferentialFunctionClassHolder.getInstance().getTensorFlowNames().entrySet()){
                String s = e.getKey();
                Integer count = tfMappedOpsImportTestCounts.get(s);
                if(count == null || count == 0){
                    log.info("TF mapped op with no import tests: {}", s);
                } else {
                    tfOpsWithImportTests++;
                }
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
        String fracLibnd4j = String.format("%.2f", 100.0 * (countLibnd4jMapped / (double)countTotalLibnd4jOps));

        String fracTFMappedTested = String.format("%.2f", 100.0 * tfOpsWithImportTests / (double)totalTFMappedOps);

        log.info("*****************************************************");
        log.info("Op Validation:                    {} of {} classes with adequate tests ({}% coverage)", countAdequate, totalFwd, pc);
        log.info("Forward pass tests:               {} of {} classes ({}% coverage)", countAdequateFwd, totalFwd, pcFwd);
        log.info("Gradient check tests:             {} of {} classes ({}% coverage)", countAdequateBwd, totalBwd, pcBwd);
        log.info("({} ops excluded from gradient check coverage)", excludedFromBackpropCoverage.size());
        log.info("({} ops excluded from fwd+gradient tests)", excludedFromAllTestCoverage.size());
        log.info("TF mapped ops:                    {} of {} ({}%)", countTfMapped, countTf, fracTfStr);
        log.info("TF mapped ops w/ import test:     {} of {} ({}%)", tfOpsWithImportTests, totalTFMappedOps, fracTFMappedTested);
        log.info("Libnd4j mapped ops:               {} of {} ({}%)", countLibnd4jMapped, countTotalLibnd4jOps, fracLibnd4j);
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
                DefaultOpConverter.class,
                EqualsWithEps.class,
                FreeGridOp.class,
                MergeSum.class, //Redundant; we use MergeAdd in samediff instead
                ScalarRemainder.class,  //Redundant; SameDiff uses ScalarFMod instead
                RestoreV2.class,
                SaveV2.class,
                ScalarSetValue.class,   //Not used in SameDiff (it's a "set to X if less than X" type op, redundant given other ops)
                LegacyPooling2D.class,  //Deprecated; not used in samediff
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

                CubeDerivative.class,
                ELUDerivative.class,
                HardSigmoidDerivative.class,
                HardTanhDerivative.class,
                LeakyReLUDerivative.class,
                LogSoftMaxDerivative.class,
                RationalTanhDerivative.class,
                RectifiedTanhDerivative.class,
                Relu6Derivative.class,
                SELUDerivative.class,
                SigmoidDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.SigmoidDerivative.class,
                SoftMaxDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.SoftMaxDerivative.class,
                SoftSignDerivative.class,
                TanhDerivative.class,
                LogSigmoidDerivative.class,
                SwishDerivative.class,
                TanDerivative.class,
                TanhDerivative.class,
                org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative.class,
                PowDerivative.class,

                BiasAddGrad.class,
                ConcatBp.class,
                TileBp.class,

                BatchNormDerivative.class,
                Conv2DDerivative.class,
                Conv3DDerivative.class,
                DeConv2DDerivative.class,
                FullConv3DDerivative.class,
                LocalResponseNormalizationDerivative.class,
                Pooling2DDerivative.class,
                Pooling3DDerivative.class,
                SConv2DDerivative.class,
                Upsampling2dDerivative.class,
                Im2colBp.class,

                SliceBp.class,
                StridedSliceBp.class,

                //We can't use these dropout ops in SameDiff: https://github.com/deeplearning4j/deeplearning4j/issues/5650
                DropOut.class,
                DropOutInverted.class,
                AlphaDropOut.class,
                Choice.class,
                ProbablisticMerge.class
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
                Histogram.class,
                InvertPermutation.class,    //Uses integer indices
                ConfusionMatrix.class,      //Integer indices
                Linspace.class,             //No input array
                //Exclude boolean operations:
                Any.class,
                All.class,
                //Exclude index accumulations (index out, not real-valued)
                FirstIndex.class,
                IAMax.class,
                IAMin.class,
                IMax.class,
                IMin.class,
                LastIndex.class,
                //Exclude Random ops
                LegacyDropOut.class,
                LegacyDropOutInverted.class,
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
                VarianceBp.class
        );

        return new HashSet<>(list);
    }

}
