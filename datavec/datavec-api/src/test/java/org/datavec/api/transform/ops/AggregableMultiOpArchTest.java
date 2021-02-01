/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.transform.ops;

import com.tngtech.archunit.core.importer.ImportOption;
import com.tngtech.archunit.junit.AnalyzeClasses;
import com.tngtech.archunit.junit.ArchTest;
import com.tngtech.archunit.junit.ArchUnitRunner;
import com.tngtech.archunit.lang.ArchRule;
import org.junit.runner.RunWith;
import org.nd4j.common.tests.BaseND4JTest;

import java.io.Serializable;

import static com.tngtech.archunit.lang.syntax.ArchRuleDefinition.classes;

/**
 * Created by dariuszzbyrad on 7/31/2020.
 */
@RunWith(ArchUnitRunner.class)
@AnalyzeClasses(packages = "org.datavec.api.transform.ops", importOptions = {ImportOption.DoNotIncludeTests.class})
public class AggregableMultiOpArchTest extends BaseND4JTest {

    @ArchTest
    public static final ArchRule ALL_AGGREGATE_OPS_MUST_BE_SERIALIZABLE = classes()
            .that().resideInAPackage("org.datavec.api.transform.ops")
            .and().doNotHaveSimpleName("AggregatorImpls")
            .and().doNotHaveSimpleName("IAggregableReduceOp")
            .and().doNotHaveSimpleName("StringAggregatorImpls")
            .and().doNotHaveFullyQualifiedName("org.datavec.api.transform.ops.StringAggregatorImpls$1")
            .should().implement(Serializable.class)
            .because("All aggregate ops must be serializable.");
}