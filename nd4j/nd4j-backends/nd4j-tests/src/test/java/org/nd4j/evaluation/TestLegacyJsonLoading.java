package org.nd4j.evaluation;

import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertEquals;

public class TestLegacyJsonLoading {

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
