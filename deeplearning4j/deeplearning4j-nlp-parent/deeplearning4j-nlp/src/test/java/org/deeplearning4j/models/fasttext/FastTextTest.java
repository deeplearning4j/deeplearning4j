package org.deeplearning4j.models.fasttext;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.resources.Resources;

import java.io.File;
import java.io.IOException;


import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;


@Slf4j
public class FastTextTest {

    private File inputFile = Resources.asFile("models/fasttext/data/labeled_data.txt");
    private File modelFile = Resources.asFile("models/fasttext/supervised.model.bin");


    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testTrainSupervised() throws IOException {

        File output = testDir.newFile();

        FastText fastText =
                 FastText.builder().supervised(true).
                 inputFile(inputFile.getAbsolutePath()).
                 outputFile(output.getAbsolutePath()).build();
        log.info("\nTraining supervised model ...\n");
        fastText.init();
        fastText.fit();
    }

    @Test
    public void testTrainSkipgram() throws IOException {

        File output = testDir.newFile();

        FastText fastText =
                FastText.builder().skipgram(true).
                        inputFile(inputFile.getAbsolutePath()).
                        outputFile(output.getAbsolutePath()).build();
        log.info("\nTraining supervised model ...\n");
        fastText.init();
        fastText.fit();
    }

    @Test
    public void testTrainSkipgramWithBuckets() throws IOException {

        File output = testDir.newFile();

        FastText fastText =
                FastText.builder().skipgram(true).
                        bucket(150).
                        inputFile(inputFile.getAbsolutePath()).
                        outputFile(output.getAbsolutePath()).build();
        log.info("\nTraining supervised model ...\n");
        fastText.init();
        fastText.fit();
    }

    @Test
    public void testTrainCBOW() throws IOException {

        File output = testDir.newFile();

        FastText fastText =
                FastText.builder().cbow(true).
                        inputFile(inputFile.getAbsolutePath()).
                        outputFile(output.getAbsolutePath()).build();
        log.info("\nTraining supervised model ...\n");
        fastText.init();
        fastText.fit();
    }

    @Test
    public void testPredict() throws IOException {
            String text = "I like soccer";

            FastText fastText = new FastText(modelFile);
            assertEquals(48, fastText.vocab().numWords());
            assertEquals("association", fastText.vocab().wordAtIndex(fastText.vocab().numWords() - 1));

            double[] expected = {-0.006423053797334433, 0.007660661358386278, 0.006068876478821039, -0.004772625397890806, -0.007143457420170307, -0.007735592778772116, -0.005607823841273785, -0.00836215727031231, 0.0011235733982175589, 2.599214785732329E-4, 0.004131870809942484, 0.007203693501651287, 0.0016768622444942594, 0.008694255724549294, -0.0012487826170399785, -0.00393667770549655, -0.006292815785855055, 0.0049359360709786415, -3.356488887220621E-4, -0.009407570585608482, -0.0026168026961386204, -0.00978928804397583, 0.0032913016621023417, -0.0029464277904480696, -0.008649969473481178, 8.056449587456882E-4, 0.0043088337406516075, -0.008980576880276203, 0.008716211654245853, 0.0073893265798687935, -0.007388216909021139, 0.003814412746578455, -0.005518500227481127, 0.004668557550758123, 0.006603693123906851, 0.003820829326286912, 0.007174000144004822, -0.006393063813447952, -0.0019381389720365405, -0.0046371882781386375, -0.006193376146256924, -0.0036685809027403593, 7.58899434003979E-4, -0.003185075242072344, -0.008330358192324638, 3.3206873922608793E-4, -0.005389622412621975, 0.009706716984510422, 0.0037855932023376226, -0.008665262721478939, -0.0032511046156287193, 4.4134497875347733E-4, -0.008377416990697384, -0.009110655635595322, 0.0019723298028111458, 0.007486093323677778, 0.006400121841579676, 0.00902814231812954, 0.00975200068205595, 0.0060582347214221954, -0.0075621469877660275, 1.0270809434587136E-4, -0.00673140911385417, -0.007316927425563335, 0.009916870854794979, -0.0011407854035496712, -4.502215306274593E-4, -0.007612560410052538, 0.008726916275918484, -3.0280642022262327E-5, 0.005529289599508047, -0.007944817654788494, 0.005593308713287115, 0.003423960180953145, 4.1348213562741876E-4, 0.009524818509817123, -0.0025129399728029966, -0.0030074280221015215, -0.007503866218030453, -0.0028124507516622543, -0.006841592025011778, -2.9375351732596755E-4, 0.007195258513092995, -0.007775942329317331, 3.951996040996164E-4, -0.006887971889227629, 0.0032655203249305487, -0.007975360378623009, -4.840183464693837E-6, 0.004651934839785099, 0.0031739831902086735, 0.004644941072911024, -0.007461248897016048, 0.003057275665923953, 0.008903342299163342, 0.006857945583760738, 0.007567950990051031, 0.001506582135334611, 0.0063307867385447025, 0.005645462777465582};
            assertArrayEquals(expected, fastText.getWordVector("association"), 1e-4);

            String label = fastText.predict(text);
            assertEquals("__label__soccer", label);
    }

    @Test
    public void testPredictProbability() throws IOException {
        String text = "I like soccer";

        FastText fastText = new FastText(modelFile);

        Pair<String,Float> result = fastText.predictProbability(text);
        assertEquals("__label__soccer", result.getFirst());
        assertEquals(-0.6930, result.getSecond(), 1e-4);

        assertEquals(48, fastText.vocabSize());
        assertEquals(0.0500, fastText.getLearningRate(), 1e-4);
        assertEquals(100, fastText.getDimension());
        assertEquals(5, fastText.getContextWindowSize());
        assertEquals(5, fastText.getEpoch());
        assertEquals(5, fastText.getNegativesNumber());
        assertEquals(1, fastText.getWordNgrams());
        assertEquals("softmax", fastText.getLossName());
        assertEquals("sup", fastText.getModelName());
        assertEquals(0, fastText.getNumberOfBuckets());
    }

    @Test
    public void testVocabulary() throws IOException {
        FastText fastText = new FastText(modelFile);
        assertEquals(48, fastText.vocab().numWords());
        assertEquals(48, fastText.vocabSize());

        String[] expected = {"</s>", ".", "is", "game", "the", "soccer", "?", "football", "3", "12", "takes", "usually", "A", "US",
        "in", "popular", "most", "hours", "and", "clubs", "minutes", "Do", "you", "like", "Is", "your", "favorite", "games",
        "Premier", "Soccer", "a", "played", "by", "two", "teams", "of", "eleven", "players", "The", "Football", "League", "an",
        "English", "professional", "league", "for", "men's", "association"};

        for (int i = 0; i < fastText.vocabSize(); ++i) {
           assertEquals(expected[i], fastText.vocab().wordAtIndex(i));
        }
    }

    @Test
    public void testLoadIterator() {
        try {
            SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());
            FastText fastText =
                    FastText.builder().supervised(true).iterator(iter).build();
            fastText.init();

        } catch (IOException e) {
            log.error(e.toString());
        }
    }

    @Test(expected=IllegalStateException.class)
    public void testState() {
        FastText fastText = new FastText();
        String label = fastText.predict("something");
    }

}
