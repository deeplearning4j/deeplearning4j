/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.nd4j.evaluation;

import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertEquals;

public class TestLegacyJsonLoading extends BaseNd4jTest {

    public TestLegacyJsonLoading(Nd4jBackend b){
        super(b);
    }

    @Override
    public char ordering(){
        return 'c';
    }

    @Test
    public void testEvalLegacyFormat() throws Exception {

        File f = new ClassPathResource("regression_testing/eval_100b/evaluation.json").getFile();
        String s = FileUtils.readFileToString(f, StandardCharsets.UTF_8);
//        System.out.println(s);

        Evaluation e = Evaluation.fromJson(s);

        assertEquals(0.78, e.accuracy(), 1e-4);
        assertEquals(0.80, e.precision(), 1e-4);
        assertEquals(0.7753, e.f1(), 1e-3);

        f = new ClassPathResource("regression_testing/eval_100b/regressionEvaluation.json").getFile();
        s = FileUtils.readFileToString(f, StandardCharsets.UTF_8);
        RegressionEvaluation re = RegressionEvaluation.fromJson(s);
        assertEquals(6.53809e-02, re.meanSquaredError(0), 1e-4);
        assertEquals(3.46236e-01, re.meanAbsoluteError(1), 1e-4);

        f = new ClassPathResource("regression_testing/eval_100b/rocMultiClass.json").getFile();
        s = FileUtils.readFileToString(f, StandardCharsets.UTF_8);
        ROCMultiClass r = ROCMultiClass.fromJson(s);

        assertEquals(0.9838, r.calculateAUC(0), 1e-4);
        assertEquals(0.7934, r.calculateAUC(1), 1e-4);
    }

}
