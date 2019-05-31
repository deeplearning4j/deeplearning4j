package org.deeplearning4j.models.fasttext;

import com.github.jfasttext.JFastText;
import org.junit.Test;

import static org.junit.Assert.assertEquals;


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

    @Test
    public void testPredict() {
        String model = "src/test/resources/models/fasttext/supervised.model.bin";
        String text = "I like soccer";

        FastText fastText = new FastText();
        fastText.loadBinaryModel(model);
        String label = fastText.predict(text);
        assertEquals("__label__soccer", label);
    }

    @Test(expected=IllegalStateException.class)
    public void testState() {
        FastText fastText = new FastText();
        String label = fastText.predict("something");
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
