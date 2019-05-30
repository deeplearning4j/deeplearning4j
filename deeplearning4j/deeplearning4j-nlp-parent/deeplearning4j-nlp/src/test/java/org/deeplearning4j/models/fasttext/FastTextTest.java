package org.deeplearning4j.models.fasttext;

import com.github.jfasttext.JFastText;
import org.junit.Test;


public class FastTextTest {


    @Test
    public void testTrainSupervised() {
        FastText.Builder builder = FastText.Builder.builder().supervised(true).
                inputFile("src/test/resources/data/labeled_data.txt").
                outputFile("src/test/resources/models/fasttext/supervised.model").build();
        FastText fastText = new FastText(builder);
        System.out.printf("\nTraining supervised model ...\n");
        fastText.fit();
    }

    // Reference test
    @Test
    public void test01TrainSupervisedCmd() {
        System.out.printf("\nTraining supervised model ...\n");
        JFastText jft = new JFastText();
        jft.runCmd(new String[] {
                "supervised",
                "-input", "src/test/resources/data/labeled_data.txt",
                "-output", "src/test/resources/models/fasttext/supervised.model"
        });
    }
}
