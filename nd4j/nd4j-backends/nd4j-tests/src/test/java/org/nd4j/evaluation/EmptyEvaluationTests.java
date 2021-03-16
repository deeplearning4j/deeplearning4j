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

package org.nd4j.evaluation;

import org.junit.jupiter.api.Test;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.EvaluationBinary;
import org.nd4j.evaluation.classification.EvaluationCalibration;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.evaluation.classification.ROCBinary;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation.Metric;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class EmptyEvaluationTests extends BaseNd4jTest {

    public EmptyEvaluationTests(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testEmptyEvaluation() {
        Evaluation e = new Evaluation();
        System.out.println(e.stats());

        for (Evaluation.Metric m : Evaluation.Metric.values()) {
            try {
                e.scoreForMetric(m);
                fail("Expected exception");
            } catch (Throwable t){
                assertTrue(t.getMessage().contains("no evaluation has been performed"),t.getMessage());
            }
        }
    }

    @Test
    public void testEmptyRegressionEvaluation() {
        RegressionEvaluation re = new RegressionEvaluation();
        re.stats();

        for (Metric m : Metric.values()) {
            try {
                re.scoreForMetric(m);
            } catch (Throwable t){
                assertTrue(t.getMessage().contains("eval must be called"),t.getMessage());
            }
        }
    }

    @Test
    public void testEmptyEvaluationBinary() {
        EvaluationBinary eb = new EvaluationBinary();
        eb.stats();

        for (EvaluationBinary.Metric m : EvaluationBinary.Metric.values()) {
            try {
                eb.scoreForMetric(m, 0);
                fail("Expected exception");
            } catch (Throwable t) {
                assertTrue( t.getMessage().contains("eval must be called"),t.getMessage());
            }
        }
    }

    @Test
    public void testEmptyROC() {
        ROC roc = new ROC();
        roc.stats();

        for (ROC.Metric m : ROC.Metric.values()) {
            try {
                roc.scoreForMetric(m);
                fail("Expected exception");
            } catch (Throwable t) {
                assertTrue(t.getMessage().contains("no evaluation"),t.getMessage());
            }
        }
    }

    @Test
    public void testEmptyROCBinary() {
        ROCBinary rb = new ROCBinary();
        rb.stats();

        for (ROCBinary.Metric m : ROCBinary.Metric.values()) {
            try {
                rb.scoreForMetric(m, 0);
                fail("Expected exception");
            } catch (Throwable t) {
                assertTrue(t.getMessage().contains("eval must be called"),t.getMessage());
            }
        }
    }

    @Test
    public void testEmptyROCMultiClass() {
        ROCMultiClass r = new ROCMultiClass();
        r.stats();

        for (ROCMultiClass.Metric m : ROCMultiClass.Metric.values()) {
            try {
                r.scoreForMetric(m, 0);
                fail("Expected exception");
            } catch (Throwable t) {
                assertTrue(t.getMessage().contains("no data"),t.getMessage());
            }
        }
    }

    @Test
    public void testEmptyEvaluationCalibration() {
        EvaluationCalibration ec = new EvaluationCalibration();
        ec.stats();

        try {
            ec.getResidualPlot(0);
            fail("Expected exception");
        } catch (Throwable t) {
            assertTrue( t.getMessage().contains("no data"),t.getMessage());
        }
        try {
            ec.getProbabilityHistogram(0);
            fail("Expected exception");
        } catch (Throwable t) {
            assertTrue( t.getMessage().contains("no data"),t.getMessage());
        }
        try {
            ec.getReliabilityDiagram(0);
            fail("Expected exception");
        } catch (Throwable t) {
            assertTrue(t.getMessage().contains("no data"),t.getMessage());
        }
    }

}
