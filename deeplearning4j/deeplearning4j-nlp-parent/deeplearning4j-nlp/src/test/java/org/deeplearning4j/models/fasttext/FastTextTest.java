package org.deeplearning4j.models.fasttext;

import com.github.jfasttext.JFastText;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.BaseDL4JTest;
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
import java.util.Arrays;


import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;


@Slf4j
public class FastTextTest extends BaseDL4JTest {

    private File inputFile = Resources.asFile("models/fasttext/data/labeled_data.txt");
    private File supModelFile = Resources.asFile("models/fasttext/supervised.model.bin");
    private File cbowModelFile = Resources.asFile("models/fasttext/cbow.model.bin");
    private File supervisedVectors = Resources.asFile("models/fasttext/supervised.model.vec");


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
        fastText.fit();
    }

    @Test
    public void tesLoadCBOWModel() throws IOException {

        FastText fastText = new FastText(cbowModelFile);
        fastText.test(cbowModelFile);

        assertEquals(19, fastText.vocab().numWords());
        assertEquals("enjoy", fastText.vocab().wordAtIndex(fastText.vocab().numWords() - 1));

        double[] expected = {5.040466203354299E-4, 0.001005030469968915, 2.8882650076411664E-4, -6.413314840756357E-4, -1.78931062691845E-4, -0.0023157168179750443, -0.002215880434960127, 0.00274421414360404, -1.5344757412094623E-4, 4.6274057240225375E-4, -1.4383681991603225E-4, 3.7832374800927937E-4, 2.523412986192852E-4, 0.0018913350068032742, -0.0024741862434893847, -4.976555937901139E-4, 0.0039220210164785385, -0.001781729981303215, -6.010578363202512E-4, -0.00244093406945467, -7.98621098510921E-4, -0.0010007203090935946, -0.001640203408896923, 7.897148607298732E-4, 9.131592814810574E-4, -0.0013367272913455963, -0.0014030139427632093, -7.755287806503475E-4, -4.2878396925516427E-4, 6.912827957421541E-4, -0.0011824817629531026, -0.0036014916840940714, 0.004353308118879795, -7.073904271237552E-5, -9.646290563978255E-4, -0.0031849315855652094, 2.3360115301329643E-4, -2.9103990527801216E-4, -0.0022990566212683916, -0.002393763978034258, -0.001034979010000825, -0.0010725988540798426, 0.0018285386031493545, -0.0013178540393710136, -1.6632364713586867E-4, -1.4665909475297667E-5, 5.445032729767263E-4, 2.999933494720608E-4, -0.0014367225812748075, -0.002345481887459755, 0.001117417006753385, -8.688368834555149E-4, -0.001830018823966384, 0.0013242220738902688, -8.880519890226424E-4, -6.888324278406799E-4, -0.0036394784692674875, 0.002179111586883664, -1.7201311129610986E-4, 0.002365073887631297, 0.002688770182430744, 0.0023955567739903927, 0.001469283364713192, 0.0011803617235273123, 5.871498142369092E-4, -7.099180947989225E-4, 7.518937345594168E-4, -8.599072461947799E-4, -6.600041524507105E-4, -0.002724145073443651, -8.365285466425121E-4, 0.0013173354091122746, 0.001083166105672717, 0.0014539906987920403, -3.1698777456767857E-4, -2.387022686889395E-4, 1.9560157670639455E-4, 0.0020277926232665777, -0.0012741144746541977, -0.0013026101514697075, -1.5212174912448972E-4, 0.0014194383984431624, 0.0012500399025157094, 0.0013362085446715355, 3.692879108712077E-4, 4.319801155361347E-5, 0.0011261265026405454, 0.0017244465416297317, 5.564604725805111E-5, 0.002170475199818611, 0.0014707016525790095, 0.001303741242736578, 0.005553730763494968, -0.0011097051901742816, -0.0013661726843565702, 0.0014100460102781653, 0.0011811562580987811, -6.622733199037611E-4, 7.860265322960913E-4, -9.811905911192298E-4};
        assertArrayEquals(expected, fastText.getWordVector("enjoy"), 1e-4);
    }

    @Test
    public void testPredict() {
            String text = "I like soccer";

            FastText fastText = new FastText(supModelFile);
            assertEquals(48, fastText.vocab().numWords());
            assertEquals("association", fastText.vocab().wordAtIndex(fastText.vocab().numWords() - 1));

            double[] expected = {-0.006423053797334433, 0.007660661358386278, 0.006068876478821039, -0.004772625397890806, -0.007143457420170307, -0.007735592778772116, -0.005607823841273785, -0.00836215727031231, 0.0011235733982175589, 2.599214785732329E-4, 0.004131870809942484, 0.007203693501651287, 0.0016768622444942594, 0.008694255724549294, -0.0012487826170399785, -0.00393667770549655, -0.006292815785855055, 0.0049359360709786415, -3.356488887220621E-4, -0.009407570585608482, -0.0026168026961386204, -0.00978928804397583, 0.0032913016621023417, -0.0029464277904480696, -0.008649969473481178, 8.056449587456882E-4, 0.0043088337406516075, -0.008980576880276203, 0.008716211654245853, 0.0073893265798687935, -0.007388216909021139, 0.003814412746578455, -0.005518500227481127, 0.004668557550758123, 0.006603693123906851, 0.003820829326286912, 0.007174000144004822, -0.006393063813447952, -0.0019381389720365405, -0.0046371882781386375, -0.006193376146256924, -0.0036685809027403593, 7.58899434003979E-4, -0.003185075242072344, -0.008330358192324638, 3.3206873922608793E-4, -0.005389622412621975, 0.009706716984510422, 0.0037855932023376226, -0.008665262721478939, -0.0032511046156287193, 4.4134497875347733E-4, -0.008377416990697384, -0.009110655635595322, 0.0019723298028111458, 0.007486093323677778, 0.006400121841579676, 0.00902814231812954, 0.00975200068205595, 0.0060582347214221954, -0.0075621469877660275, 1.0270809434587136E-4, -0.00673140911385417, -0.007316927425563335, 0.009916870854794979, -0.0011407854035496712, -4.502215306274593E-4, -0.007612560410052538, 0.008726916275918484, -3.0280642022262327E-5, 0.005529289599508047, -0.007944817654788494, 0.005593308713287115, 0.003423960180953145, 4.1348213562741876E-4, 0.009524818509817123, -0.0025129399728029966, -0.0030074280221015215, -0.007503866218030453, -0.0028124507516622543, -0.006841592025011778, -2.9375351732596755E-4, 0.007195258513092995, -0.007775942329317331, 3.951996040996164E-4, -0.006887971889227629, 0.0032655203249305487, -0.007975360378623009, -4.840183464693837E-6, 0.004651934839785099, 0.0031739831902086735, 0.004644941072911024, -0.007461248897016048, 0.003057275665923953, 0.008903342299163342, 0.006857945583760738, 0.007567950990051031, 0.001506582135334611, 0.0063307867385447025, 0.005645462777465582};
            assertArrayEquals(expected, fastText.getWordVector("association"), 1e-4);

            String label = fastText.predict(text);
            assertEquals("__label__soccer", label);
    }

    @Test
    public void testPredictProbability() {
        String text = "I like soccer";

        FastText fastText = new FastText(supModelFile);

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
        FastText fastText = new FastText(supModelFile);
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
            fastText.loadIterator();

        } catch (IOException e) {
            log.error(e.toString());
        }
    }

    @Test(expected=IllegalStateException.class)
    public void testState() {
        FastText fastText = new FastText();
        String label = fastText.predict("something");
    }

    @Test
    public void testPretrainedVectors() throws IOException {
        File output = testDir.newFile();

        FastText fastText =
                FastText.builder().supervised(true).
                        inputFile(inputFile.getAbsolutePath()).
                        pretrainedVectorsFile(supervisedVectors.getAbsolutePath()).
                        outputFile(output.getAbsolutePath()).build();
        log.info("\nTraining supervised model ...\n");
        fastText.fit();
    }

    @Test
    public void testWordsStatistics() throws IOException {

        File output = testDir.newFile();

        FastText fastText =
                FastText.builder().supervised(true).
                        inputFile(inputFile.getAbsolutePath()).
                        outputFile(output.getAbsolutePath()).build();

        log.info("\nTraining supervised model ...\n");
        fastText.fit();

        Word2Vec word2Vec = WordVectorSerializer.readAsCsv(new File(output.getAbsolutePath() + ".vec"));

        assertEquals(48,  word2Vec.getVocab().numWords());

        System.out.println(word2Vec.wordsNearest("association", 3));
        System.out.println(word2Vec.similarity("Football", "teams"));
        System.out.println(word2Vec.similarity("professional", "minutes"));
        System.out.println(word2Vec.similarity("java","cpp"));
    }


    @Test
    public void testWordsNativeStatistics() throws IOException {

        File output = testDir.newFile();

        FastText fastText = new FastText();
        fastText.loadPretrainedVectors(supervisedVectors);

        log.info("\nTraining supervised model ...\n");

        assertEquals(48,  fastText.vocab().numWords());

        String[] result = new String[3];
        fastText.wordsNearest("association", 3).toArray(result);
        assertArrayEquals(new String[]{"most","eleven","hours"}, result);
        assertEquals(0.1657, fastText.similarity("Football", "teams"), 1e-4);
        assertEquals(0.3661, fastText.similarity("professional", "minutes"), 1e-4);
        assertEquals(Double.NaN, fastText.similarity("java","cpp"), 1e-4);
    }
}
