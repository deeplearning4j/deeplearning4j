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

package org.nd4j;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.nd4j.autodiff.opvalidation.*;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.imports.TFGraphs.TFGraphTestAllLibnd4j;
import org.nd4j.imports.TFGraphs.TFGraphTestAllSameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Function;

import static org.junit.Assume.assumeFalse;

/**
 * Op validation test suite.
 * Should include all classes using the {@link OpValidation} test framework, so test coverage can be calculated and reported.
 *
 * NOTE: For the op coverage information to work properly, we need the op validation to be run via this suite.
 * Otherwise, we could report coverage information before all test have run - underestimating coverage.
 *
 * For op coverage information to be collected, you need to execute a test like:<br>
 * SINGLE OP TEST: OpValidation.validate(new OpTestCase(op).expectedOutputs(0, <INDArray here>))
 *     - OpTestCase checks the output values of a single op, no backprop/gradients<br>
 *     - Returns an error message if op failed, or NULL if op passed<br>
 * SAMEDIFF TEST:  OpValidation.validate(new TestCase(sameDiff).gradientCheck(true).expectedOutput("someVar", <INDArray>))<br>
 *     - These tests can be used to check both gradients AND expected output, collecting coverage as required<br>
 *     - Returns an error message if op failed, or NULL if op passed<br>
 *     - Note gradientCheck(true) is the default<br>
 *     - Expected outputs are optional<br>
 *     - You can specify a function for validating the correctness of each output using {@link org.nd4j.autodiff.validation.TestCase#expected(String, Function)}<br>
 *
 */
@RunWith(Suite.class)
@Suite.SuiteClasses({
        //Note: these will be run as part of the suite only, and will NOT be run again separately
        LayerOpValidation.class,
        LossOpValidation.class,
        MiscOpValidation.class,
        RandomOpValidation.class,
        ReductionBpOpValidation.class,
        ReductionOpValidation.class,
        ShapeOpValidation.class,
        TransformOpValidation.class,

        //TF import tests
        TFGraphTestAllSameDiff.class,
        TFGraphTestAllLibnd4j.class
})
public class OpValidationSuite {

    /*
    Change this variable from false to true to ignore any tests that call OpValidationSuite.ignoreFailing()

    The idea: failing SameDiff tests are disabled by default, but can be re-enabled.
    This is so we can prevent regressions on already passing tests
     */
    public static final boolean IGNORE_FAILING = true;

    public static void ignoreFailing(){
        //If IGNORE_FAILING
        assumeFalse(IGNORE_FAILING);
    }


    private static DataType initialType;

    @BeforeClass
    public static void beforeClass() throws Exception {
        Nd4j.create(1);
        initialType = Nd4j.dataType();

        Nd4j.setDataType(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
    }

    @AfterClass
    public static void afterClass() throws Exception {
        Nd4j.setDataType(initialType);

        // Log coverage information
        OpValidation.logCoverageInformation(true, true, true, true, true);
    }






}
