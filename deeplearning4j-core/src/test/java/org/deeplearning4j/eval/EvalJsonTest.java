package org.deeplearning4j.eval;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.eval.curves.Histogram;
import org.deeplearning4j.eval.curves.PrecisionRecallCurve;
import org.deeplearning4j.eval.curves.RocCurve;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.TestCase.assertNull;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;


public class EvalJsonTest extends BaseDL4JTest {

    @Test
    public void testSerdeEmpty() {
        boolean print = false;

        IEvaluation[] arr = new IEvaluation[] {new Evaluation(), new EvaluationBinary(), new ROCBinary(10),
                        new ROCMultiClass(10), new RegressionEvaluation(3), new RegressionEvaluation(),
                        new EvaluationCalibration()};

        for (IEvaluation e : arr) {
            String json = e.toJson();
            String stats = e.stats();
            if (print) {
                System.out.println(e.getClass() + "\n" + json + "\n\n");
            }

            IEvaluation fromJson = BaseEvaluation.fromJson(json, BaseEvaluation.class);
            assertEquals(e.toJson(), fromJson.toJson());
        }
    }

    @Test
    public void testSerde() {
        boolean print = true;
        Nd4j.getRandom().setSeed(12345);

        Evaluation evaluation = new Evaluation();
        EvaluationBinary evaluationBinary = new EvaluationBinary();
        ROC roc = new ROC(2);
        ROCBinary roc2 = new ROCBinary(2);
        ROCMultiClass roc3 = new ROCMultiClass(2);
        RegressionEvaluation regressionEvaluation = new RegressionEvaluation();
        EvaluationCalibration ec = new EvaluationCalibration();


        IEvaluation[] arr = new IEvaluation[] {evaluation, evaluationBinary, roc, roc2, roc3, regressionEvaluation, ec};

        INDArray evalLabel = Nd4j.create(10, 3);
        for (int i = 0; i < 10; i++) {
            evalLabel.putScalar(i, i % 3, 1.0);
        }
        INDArray evalProb = Nd4j.rand(10, 3);
        evalProb.diviColumnVector(evalProb.sum(1));
        evaluation.eval(evalLabel, evalProb);
        roc3.eval(evalLabel, evalProb);
        ec.eval(evalLabel, evalProb);

        evalLabel = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(10, 3), 0.5));
        evalProb = Nd4j.rand(10, 3);
        evaluationBinary.eval(evalLabel, evalProb);
        roc2.eval(evalLabel, evalProb);

        evalLabel = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(10, 1), 0.5));
        evalProb = Nd4j.rand(10, 1);
        roc.eval(evalLabel, evalProb);

        regressionEvaluation.eval(Nd4j.rand(10, 3), Nd4j.rand(10, 3));



        for (IEvaluation e : arr) {
            String json = e.toJson();
            if (print) {
                System.out.println(e.getClass() + "\n" + json + "\n\n");
            }

            IEvaluation fromJson = BaseEvaluation.fromJson(json, BaseEvaluation.class);
            assertEquals(e.toJson(), fromJson.toJson());
        }
    }

    @Test
    public void testSerdeExactRoc() {
        Nd4j.getRandom().setSeed(12345);
        boolean print = true;

        ROC roc = new ROC(0);
        ROCBinary roc2 = new ROCBinary(0);
        ROCMultiClass roc3 = new ROCMultiClass(0);


        IEvaluation[] arr = new IEvaluation[] {roc, roc2, roc3};

        INDArray evalLabel = Nd4j.create(100, 3);
        for (int i = 0; i < 100; i++) {
            evalLabel.putScalar(i, i % 3, 1.0);
        }
        INDArray evalProb = Nd4j.rand(100, 3);
        evalProb.diviColumnVector(evalProb.sum(1));
        roc3.eval(evalLabel, evalProb);

        evalLabel = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(100, 3), 0.5));
        evalProb = Nd4j.rand(100, 3);
        roc2.eval(evalLabel, evalProb);

        evalLabel = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(100, 1), 0.5));
        evalProb = Nd4j.rand(100, 1);
        roc.eval(evalLabel, evalProb);

        for (IEvaluation e : arr) {
            System.out.println(e.getClass());
            String json = e.toJson();
            String stats = e.stats();
            if (print) {
                System.out.println(json + "\n\n");
            }
            IEvaluation fromJson = BaseEvaluation.fromJson(json, BaseEvaluation.class);
            assertEquals(e, fromJson);

            if (fromJson instanceof ROC) {
                //Shouldn't have probAndLabel, but should have stored AUC and AUPRC
                assertNull(((ROC) fromJson).getProbAndLabel());
                assertTrue(((ROC) fromJson).calculateAUC() > 0.0);
                assertTrue(((ROC) fromJson).calculateAUCPR() > 0.0);

                assertEquals(((ROC) e).getRocCurve(), ((ROC) fromJson).getRocCurve());
                assertEquals(((ROC) e).getPrecisionRecallCurve(), ((ROC) fromJson).getPrecisionRecallCurve());
            } else if (e instanceof ROCBinary) {
                ROC[] rocs = ((ROCBinary) fromJson).getUnderlying();
                ROC[] origRocs = ((ROCBinary) e).getUnderlying();
                //                for(ROC r : rocs ){
                for (int i = 0; i < origRocs.length; i++) {
                    ROC r = rocs[i];
                    ROC origR = origRocs[i];
                    //Shouldn't have probAndLabel, but should have stored AUC and AUPRC, AND stored curves
                    assertNull(r.getProbAndLabel());
                    assertEquals(origR.calculateAUC(), origR.calculateAUC(), 1e-6);
                    assertEquals(origR.calculateAUCPR(), origR.calculateAUCPR(), 1e-6);
                    assertEquals(origR.getRocCurve(), origR.getRocCurve());
                    assertEquals(origR.getPrecisionRecallCurve(), origR.getPrecisionRecallCurve());
                }

            } else if (e instanceof ROCMultiClass) {
                ROC[] rocs = ((ROCMultiClass) fromJson).getUnderlying();
                ROC[] origRocs = ((ROCMultiClass) e).getUnderlying();
                for (int i = 0; i < origRocs.length; i++) {
                    ROC r = rocs[i];
                    ROC origR = origRocs[i];
                    //Shouldn't have probAndLabel, but should have stored AUC and AUPRC, AND stored curves
                    assertNull(r.getProbAndLabel());
                    assertEquals(origR.calculateAUC(), origR.calculateAUC(), 1e-6);
                    assertEquals(origR.calculateAUCPR(), origR.calculateAUCPR(), 1e-6);
                    assertEquals(origR.getRocCurve(), origR.getRocCurve());
                    assertEquals(origR.getPrecisionRecallCurve(), origR.getPrecisionRecallCurve());
                }
            }
        }
    }

    @Test
    public void testJsonYamlCurves() {
        ROC roc = new ROC(0);

        INDArray evalLabel =
                        Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.createUninitialized(100, 1), 0.5));
        INDArray evalProb = Nd4j.rand(100, 1);
        roc.eval(evalLabel, evalProb);

        RocCurve c = roc.getRocCurve();
        PrecisionRecallCurve prc = roc.getPrecisionRecallCurve();

        String json1 = c.toJson();
        String json2 = prc.toJson();

        RocCurve c2 = RocCurve.fromJson(json1);
        PrecisionRecallCurve prc2 = PrecisionRecallCurve.fromJson(json2);

        assertEquals(c, c2);
        assertEquals(prc, prc2);

        //        System.out.println(json1);

        //Also test: histograms

        EvaluationCalibration ec = new EvaluationCalibration();

        evalLabel = Nd4j.create(10, 3);
        for (int i = 0; i < 10; i++) {
            evalLabel.putScalar(i, i % 3, 1.0);
        }
        evalProb = Nd4j.rand(10, 3);
        evalProb.diviColumnVector(evalProb.sum(1));
        ec.eval(evalLabel, evalProb);

        Histogram[] histograms = new Histogram[] {ec.getResidualPlotAllClasses(), ec.getResidualPlot(0),
                        ec.getResidualPlot(1), ec.getProbabilityHistogramAllClasses(), ec.getProbabilityHistogram(0),
                        ec.getProbabilityHistogram(1)};

        for (Histogram h : histograms) {
            String json = h.toJson();
            String yaml = h.toYaml();

            Histogram h2 = Histogram.fromJson(json);
            Histogram h3 = Histogram.fromYaml(yaml);

            assertEquals(h, h2);
            assertEquals(h2, h3);
        }

    }

    @Test
    public void testJsonWithCustomThreshold() {

        //Evaluation - binary threshold
        Evaluation e = new Evaluation(0.25);
        String json = e.toJson();
        String yaml = e.toYaml();

        Evaluation eFromJson = Evaluation.fromJson(json);
        Evaluation eFromYaml = Evaluation.fromYaml(yaml);

        assertEquals(0.25, eFromJson.getBinaryDecisionThreshold(), 1e-6);
        assertEquals(0.25, eFromYaml.getBinaryDecisionThreshold(), 1e-6);


        //Evaluation: custom cost array
        INDArray costArray = Nd4j.create(new double[] {1.0, 2.0, 3.0});
        Evaluation e2 = new Evaluation(costArray);

        json = e2.toJson();
        yaml = e2.toYaml();

        eFromJson = Evaluation.fromJson(json);
        eFromYaml = Evaluation.fromYaml(yaml);

        assertEquals(costArray, eFromJson.getCostArray());
        assertEquals(costArray, eFromYaml.getCostArray());



        //EvaluationBinary - per-output binary threshold
        INDArray threshold = Nd4j.create(new double[] {1.0, 0.5, 0.25});
        EvaluationBinary eb = new EvaluationBinary(threshold);

        json = eb.toJson();
        yaml = eb.toYaml();

        EvaluationBinary ebFromJson = EvaluationBinary.fromJson(json);
        EvaluationBinary ebFromYaml = EvaluationBinary.fromYaml(yaml);

        assertEquals(threshold, ebFromJson.getDecisionThreshold());
        assertEquals(threshold, ebFromYaml.getDecisionThreshold());

    }

}
