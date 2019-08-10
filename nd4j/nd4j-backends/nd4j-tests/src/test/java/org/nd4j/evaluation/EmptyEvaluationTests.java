package org.nd4j.evaluation;

import org.junit.Test;
import org.nd4j.evaluation.classification.*;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation.Metric;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

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
                assertTrue(t.getMessage(), t.getMessage().contains("no evaluation has been performed"));
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
                assertTrue(t.getMessage(), t.getMessage().contains("eval must be called"));
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
                assertTrue(t.getMessage(), t.getMessage().contains("eval must be called"));
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
                assertTrue(t.getMessage(), t.getMessage().contains("no evaluation"));
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
                assertTrue(t.getMessage(), t.getMessage().contains("eval must be called"));
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
                assertTrue(t.getMessage(), t.getMessage().contains("no data"));
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
            assertTrue(t.getMessage(), t.getMessage().contains("no data"));
        }
        try {
            ec.getProbabilityHistogram(0);
            fail("Expected exception");
        } catch (Throwable t) {
            assertTrue(t.getMessage(), t.getMessage().contains("no data"));
        }
        try {
            ec.getReliabilityDiagram(0);
            fail("Expected exception");
        } catch (Throwable t) {
            assertTrue(t.getMessage(), t.getMessage().contains("no data"));
        }
    }

}
