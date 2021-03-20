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

package org.datavec.api.transform.serde;

import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.serde.testClasses.CustomCondition;
import org.datavec.api.transform.serde.testClasses.CustomFilter;
import org.datavec.api.transform.serde.testClasses.CustomTransform;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.common.tests.tags.TagNames;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Tag(TagNames.JAVA_ONLY)
@Tag(TagNames.FILE_IO)
@Tag(TagNames.JACKSON_SERDE)
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
public class TestCustomTransformJsonYaml extends BaseND4JTest {

    @Test
    public void testCustomTransform() {

        Schema schema = new Schema.Builder().addColumnInteger("firstCol").addColumnDouble("secondCol").build();

        TransformProcess tp = new TransformProcess.Builder(schema).integerMathOp("firstCol", MathOp.Add, 1)
                        .transform(new CustomTransform("secondCol", 3.14159))
                        .doubleMathOp("secondCol", MathOp.Multiply, 2.0).filter(new CustomFilter(123))
                        .filter(new CustomCondition("someArg")).build();

        String asJson = tp.toJson();
        String asYaml = tp.toYaml();

        TransformProcess fromJson = TransformProcess.fromJson(asJson);
        TransformProcess fromYaml = TransformProcess.fromYaml(asYaml);

        assertEquals(tp, fromJson);
        assertEquals(tp, fromYaml);
    }

}
