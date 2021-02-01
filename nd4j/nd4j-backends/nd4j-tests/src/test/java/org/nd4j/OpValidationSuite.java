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

package org.nd4j;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.nd4j.autodiff.opvalidation.*;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.imports.tfgraphs.TFGraphTestAllSameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.function.Function;

import static org.junit.Assume.assumeFalse;

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
        TFGraphTestAllSameDiff.class
        //TFGraphTestAllLibnd4j.class
})
//IMPORTANT: This ignore is added to avoid maven surefire running both the suite AND the individual tests in "mvn test"
// With it ignored here, the individual tests will run outside (i.e., separately/independently) of the suite in both "mvn test" and IntelliJ
@Ignore
public class OpValidationSuite {

    /*
    Change this variable from false to true to ignore any tests that call OpValidationSuite.ignoreFailing()

    The idea: failing SameDiff tests are disabled by default, but can be re-enabled.
    This is so we can prevent regressions on already passing tests
     */
    public static final boolean IGNORE_FAILING = true;

    /**
     * NOTE: Do not change this.
     * If all tests won't run,
     * it's likely because of a mis specified test name.
     * Keep this trigger as is for ignoring tests.
     */
    public static void ignoreFailing() {
        //If IGNORE_FAILING
        assumeFalse(IGNORE_FAILING);
    }


    private static DataType initialType;

    @BeforeClass
    public static void beforeClass() {
        Nd4j.create(1);
        initialType = Nd4j.dataType();

        Nd4j.setDataType(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
    }

    @AfterClass
    public static void afterClass() {
        Nd4j.setDataType(initialType);

        // Log coverage information
        OpValidation.logCoverageInformation(true, true, true, true, true);
    }






}
