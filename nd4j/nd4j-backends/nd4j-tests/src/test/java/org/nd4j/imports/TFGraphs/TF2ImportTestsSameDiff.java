/* ******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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

package org.nd4j.imports.TFGraphs;

import lombok.extern.slf4j.Slf4j;
import org.junit.*;
import org.junit.rules.TestWatcher;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.OpValidationSuite;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

@Slf4j
@RunWith(Parameterized.class)
@Ignore //AB 2020/05/12 - Disabled until TF 2.x import test resources are available
public class TF2ImportTestsSameDiff {   //Note: Can't extend BaseNd4jTest here as we need no-arg constructor for parameterized tests

    @Rule
    public TestWatcher testWatcher = new TestWatcher() {

        @Override
        protected void starting(Description description){
            log.info("TF2ImportTestsSameDiff: Starting parameterized test: " + description.getDisplayName());
        }

        //protected void failed(Throwable e, Description description) {
        //protected void succeeded(Description description) {
    };

    private String modelName;
    private TestCase testCase;
    private String baseDir;

    private static final TFGraphTestAllHelper.ExecuteWith EXECUTE_WITH = TFGraphTestAllHelper.ExecuteWith.SAMEDIFF;
    public static final String[] BASE_DIRS = new String[]{"tf_graphs/examples2.1"};        //Add directories for any other TensorFlow versions here
    public static final String MODEL_FILENAME = "frozen_model.pb";

    public static final String[] IGNORE_REGEXES = new String[]{

    };

    /* As per TFGraphTestList.printArraysDebugging - this field defines a set of regexes for test cases that should have
       all arrays printed during execution.
       If a test name matches any regex here, an ExecPrintListener will be added to the listeners, and all output
       arrays will be printed during execution
     */
    private final List<String> debugModeRegexes = null; //Arrays.asList("resize_nearest_neighbor/.*", "add_n.*");

    @BeforeClass
    public static void beforeClass() {
        Nd4j.setDataType(DataType.FLOAT);
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
    }

    @Before
    public void setup() {
        Nd4j.setDataType(DataType.FLOAT);
        Nd4j.getExecutioner().enableDebugMode(false);
        Nd4j.getExecutioner().enableVerboseMode(false);
    }

    @Parameterized.Parameters(name="{3}")
    public static Collection<Object[]> data() throws Exception {
        List<Object[]> out = new ArrayList<>();

        for(String dir : BASE_DIRS) {
            String version = dir.replaceAll("tf_graphs/examples", "tf");
            Map<String, TestCase> m = TFGraphUtil.getTestCases(dir, false);
            List<String> l = new ArrayList<>(m.keySet());
            Collections.sort(l);
            for (String s : l) {
                out.add(new Object[]{s, m.get(s), dir, version + "/" + s});
            }
        }
        return out;
    }

    public TF2ImportTestsSameDiff(String name, TestCase tc, String baseDir, String displayName){
        this.modelName = name;
        this.testCase = tc;
        this.baseDir = baseDir;
    }

    @Test
    public void testOutputOnly() throws Exception {
        if(TFGraphTestZooModels.isPPC()){
            /*
            Ugly hack to temporarily disable tests on PPC only on CI
            Issue logged here: https://github.com/deeplearning4j/deeplearning4j/issues/7657
            These will be re-enabled for PPC once fixed - in the mean time, remaining tests will be used to detect and prevent regressions
             */

            log.warn("TEMPORARILY SKIPPING TEST ON PPC ARCHITECTURE DUE TO KNOWN JVM CRASH ISSUES - SEE https://github.com/deeplearning4j/deeplearning4j/issues/7657");
            OpValidationSuite.ignoreFailing();
        }


        Nd4j.create(1);

        for(String s : IGNORE_REGEXES){
            if(modelName.matches(s)){
                log.info("\n\tIGNORE MODEL ON REGEX: {} - regex {}", modelName, s);
                OpValidationSuite.ignoreFailing();
            }
        }
        Pair<Double,Double> precisionOverride = TFGraphTestAllHelper.testPrecisionOverride(modelName);
        Double maxRE = (precisionOverride == null ? null : precisionOverride.getFirst());
        Double minAbs = (precisionOverride == null ? null : precisionOverride.getSecond());

        boolean verboseDebugMode = false;
        if(debugModeRegexes != null){
            for(String regex : debugModeRegexes){
                if(modelName.matches(regex)){
                    verboseDebugMode = true;
                    break;
                }
            }
        }

        Map<String,INDArray> inputs = TFGraphUtil.loadInputs(testCase);
        Map<String,INDArray> predictions = TFGraphUtil.loadPredictions(testCase);


        try {
            TFGraphTestAllHelper.checkOnlyOutput(inputs, predictions, modelName, baseDir, MODEL_FILENAME, EXECUTE_WITH, TFGraphTestAllHelper.LOADER, maxRE, minAbs, verboseDebugMode);
            //TFGraphTestAllHelper.checkIntermediate(inputs, modelName, BASE_DIR, MODEL_FILENAME, EXECUTE_WITH, localTestDir);
        } catch (Throwable t){
            log.error("ERROR Executing test: {} - input keys {}", modelName, (inputs == null ? null : inputs.keySet()), t);
            throw t;
        }
        //TFGraphTestAllHelper.checkIntermediate(inputs, modelName, EXECUTE_WITH);
    }

}
