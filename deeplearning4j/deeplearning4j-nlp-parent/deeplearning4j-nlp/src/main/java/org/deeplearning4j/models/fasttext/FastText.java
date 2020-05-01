package org.deeplearning4j.models.fasttext;

import com.github.jfasttext.JFastText;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

import java.io.*;
import java.util.*;

@Slf4j
@AllArgsConstructor
@lombok.Builder
public class FastText implements WordVectors, Serializable {

    private final static String METHOD_NOT_AVAILABLE = "This method is available for text (.vec) models only - binary (.bin) model currently loaded";
    // Mandatory
    @Getter private String inputFile;
    @Getter private String outputFile;

    // Optional for dictionary
    @Builder.Default  private int bucket = -1;
    @Builder.Default  private int minCount = -1;
    @Builder.Default  private int minCountLabel = -1;
    @Builder.Default  private int wordNgrams = -1;
    @Builder.Default  private int minNgramLength = -1;
    @Builder.Default  private int maxNgramLength = -1;
    @Builder.Default  private int samplingThreshold = -1;
    private String labelPrefix;

    // Optional for training
    @Getter private boolean supervised;
    @Getter private boolean quantize;
    @Getter private boolean predict;
    @Getter private boolean predict_prob;
    @Getter private boolean skipgram;
    @Getter private boolean cbow;
    @Getter private boolean nn;
    @Getter private boolean analogies;
    @Getter private String pretrainedVectorsFile;
    @Getter
    @Builder.Default
    private double learningRate = -1.0;
    @Getter private double learningRateUpdate = -1.0;
    @Getter
    @Builder.Default
    private int dim = -1;
    @Getter
    @Builder.Default
    private int contextWindowSize = -1;
    @Getter
    @Builder.Default
    private int epochs = -1;
    @Getter private String modelName;
    @Getter private String lossName;
    @Getter
    @Builder.Default
    private int negativeSamples = -1;
    @Getter
    @Builder.Default
    private int numThreads = -1;
    @Getter private boolean saveOutput = false;

    // Optional for quantization
    @Getter
    @Builder.Default
    private int cutOff = -1;
    @Getter private boolean retrain;
    @Getter private boolean qnorm;
    @Getter private boolean qout;
    @Getter
    @Builder.Default
    private int dsub = -1;

    @Getter private SentenceIterator iterator;

    @Builder.Default private transient JFastText fastTextImpl = new JFastText();
    private transient Word2Vec word2Vec;
    @Getter private boolean modelLoaded;
    @Getter private boolean modelVectorsLoaded;
    private VocabCache vocabCache;

    public FastText(File modelPath) {
        this();
        loadBinaryModel(modelPath.getAbsolutePath());
    }

    public FastText() {
        fastTextImpl = new JFastText();
    }

    private static class ArgsFactory {

        private List<String> args = new ArrayList<>();

        private void add(String label, String value) {
            args.add(label);
            args.add(value);
        }

        private void addOptional(String label, int value) {
            if (value >= 0) {
                args.add(label);
                args.add(Integer.toString(value));
            }
        }

        private void addOptional(String label, double value) {
            if (value >= 0.0) {
                args.add(label);
                args.add(Double.toString(value));
            }
        }

        private void addOptional(String label, String value) {
            if (StringUtils.isNotEmpty(value)) {
                args.add(label);
                args.add(value);
            }
        }

        private void addOptional(String label, boolean value) {
            if (value) {
                args.add(label);
            }
        }


        public String[] args() {
            String[] asArray = new String[args.size()];
            return args.toArray(asArray);
        }
    }

    private String[] makeArgs() {
        ArgsFactory argsFactory = new ArgsFactory();

        argsFactory.addOptional("cbow", cbow);
        argsFactory.addOptional("skipgram", skipgram);
        argsFactory.addOptional("supervised", supervised);
        argsFactory.addOptional("quantize", quantize);
        argsFactory.addOptional("predict", predict);
        argsFactory.addOptional("predict_prob", predict_prob);

        argsFactory.add("-input", inputFile);
        argsFactory.add("-output", outputFile );

        argsFactory.addOptional("-pretrainedVectors", pretrainedVectorsFile);

        argsFactory.addOptional("-bucket", bucket);
        argsFactory.addOptional("-minCount", minCount);
        argsFactory.addOptional("-minCountLabel", minCountLabel);
        argsFactory.addOptional("-wordNgrams", wordNgrams);
        argsFactory.addOptional("-minn", minNgramLength);
        argsFactory.addOptional("-maxn", maxNgramLength);
        argsFactory.addOptional("-t", samplingThreshold);
        argsFactory.addOptional("-label", labelPrefix);
        argsFactory.addOptional("analogies",analogies);
        argsFactory.addOptional("-lr", learningRate);
        argsFactory.addOptional("-lrUpdateRate", learningRateUpdate);
        argsFactory.addOptional("-dim", dim);
        argsFactory.addOptional("-ws", contextWindowSize);
        argsFactory.addOptional("-epoch", epochs);
        argsFactory.addOptional("-loss", lossName);
        argsFactory.addOptional("-neg", negativeSamples);
        argsFactory.addOptional("-thread", numThreads);
        argsFactory.addOptional("-saveOutput", saveOutput);
        argsFactory.addOptional("-cutoff", cutOff);
        argsFactory.addOptional("-retrain", retrain);
        argsFactory.addOptional("-qnorm", qnorm);
        argsFactory.addOptional("-qout", qout);
        argsFactory.addOptional("-dsub", dsub);

        return argsFactory.args();
    }

    public void fit() {
        String[] cmd = makeArgs();
        fastTextImpl.runCmd(cmd);
    }

    public void loadIterator() {
        if (iterator != null) {
            try {
                File tempFile = File.createTempFile("FTX", ".txt");
                BufferedWriter writer = new BufferedWriter(new FileWriter(tempFile));
                while (iterator.hasNext()) {
                    String sentence = iterator.nextSentence();
                    writer.write(sentence);
                }

                fastTextImpl = new JFastText();
            } catch (IOException e) {
                log.error(e.getMessage());
            }
        }
    }

    public void loadPretrainedVectors(File vectorsFile) {
        word2Vec = WordVectorSerializer.readWord2VecModel(vectorsFile);
        modelVectorsLoaded = true;
        log.info("Loaded vectorized representation from file %s. Functionality will be restricted.",
                vectorsFile.getAbsolutePath());
    }

    public void loadBinaryModel(String modelPath) {
        fastTextImpl.loadModel(modelPath);

        modelLoaded = true;
    }

    public void unloadBinaryModel() {
        fastTextImpl.unloadModel();
        modelLoaded = false;
    }

    public void test(File testFile) {
        fastTextImpl.test(testFile.getAbsolutePath());
    }

    private void assertModelLoaded() {
        if (!modelLoaded && !modelVectorsLoaded)
            throw new IllegalStateException("Model must be loaded before predict!");
    }

    public String predict(String text) {

        assertModelLoaded();

        String label = fastTextImpl.predict(text);
        return label;
    }

    public Pair<String, Float> predictProbability(String text) {

        assertModelLoaded();

        JFastText.ProbLabel predictedProbLabel = fastTextImpl.predictProba(text);

        Pair<String,Float> retVal = new Pair<>();
        retVal.setFirst(predictedProbLabel.label);
        retVal.setSecond(predictedProbLabel.logProb);
        return retVal;
    }

    @Override
    public VocabCache vocab() {
        if (modelVectorsLoaded) {
            vocabCache = word2Vec.vocab();
        }
        else {
            if (!modelLoaded)
                throw new IllegalStateException("Load model before calling vocab()");

            if (vocabCache == null) {
                vocabCache = new AbstractCache();
            }
            List<String> words = fastTextImpl.getWords();
            for (int i = 0; i < words.size(); ++i) {
                vocabCache.addWordToIndex(i, words.get(i));
                VocabWord word = new VocabWord();
                word.setWord(words.get(i));
                vocabCache.addToken(word);
            }
        }
        return vocabCache;
    }

    @Override
    public long vocabSize() {
        long result = 0;
        if (modelVectorsLoaded) {
            result = word2Vec.vocabSize();
        }
        else {
            if (!modelLoaded)
                throw new IllegalStateException("Load model before calling vocab()");
            result = fastTextImpl.getNWords();
        }
        return result;
    }

    @Override
    public String getUNK() {
        throw new NotImplementedException("FastText.getUNK");
    }

    @Override
    public void setUNK(String input) {
        throw new NotImplementedException("FastText.setUNK");
    }

    @Override
    public double[] getWordVector(String word) {
        if (modelVectorsLoaded) {
            return word2Vec.getWordVector(word);
        }
        else {
            List<Float> vectors = fastTextImpl.getVector(word);
            double[] retVal = new double[vectors.size()];
            for (int i = 0; i < vectors.size(); ++i) {
                retVal[i] = vectors.get(i);
            }
            return retVal;
        }
    }

    @Override
    public INDArray getWordVectorMatrixNormalized(String word) {
        if (modelVectorsLoaded) {
            return word2Vec.getWordVectorMatrixNormalized(word);
        }
        else {
            INDArray r = getWordVectorMatrix(word);
            return r.divi(Nd4j.getBlasWrapper().nrm2(r));
        }
    }

    @Override
    public INDArray getWordVectorMatrix(String word) {
        if (modelVectorsLoaded) {
            return word2Vec.getWordVectorMatrix(word);
        }
        else {
            double[] values = getWordVector(word);
            return Nd4j.createFromArray(values);
        }
    }

    @Override
    public INDArray getWordVectors(Collection<String> labels) {
        if (modelVectorsLoaded) {
            return word2Vec.getWordVectors(labels);
        }
        return null;
    }

    @Override
    public INDArray getWordVectorsMean(Collection<String> labels) {
        if (modelVectorsLoaded) {
            return word2Vec.getWordVectorsMean(labels);
        }
        return null;
    }

    private List<String> words = new ArrayList<>();

    @Override
    public boolean hasWord(String word) {
        if (modelVectorsLoaded) {
            return word2Vec.outOfVocabularySupported();
        }
        if (words.isEmpty())
            words = fastTextImpl.getWords();
        return words.contains(word);
    }

    @Override
    public Collection<String> wordsNearest(INDArray words, int top) {
        if (modelVectorsLoaded) {
            return word2Vec.wordsNearest(words, top);
        }
        throw new IllegalStateException(METHOD_NOT_AVAILABLE);
    }

    @Override
    public Collection<String> wordsNearestSum(INDArray words, int top) {
        if (modelVectorsLoaded) {
            return word2Vec.wordsNearestSum(words, top);
        }
        throw new IllegalStateException(METHOD_NOT_AVAILABLE);
    }

    @Override
    public Collection<String> wordsNearestSum(String word, int n) {
        if (modelVectorsLoaded) {
            return word2Vec.wordsNearestSum(word, n);
        }
        throw new IllegalStateException(METHOD_NOT_AVAILABLE);
    }


    @Override
    public Collection<String> wordsNearestSum(Collection<String> positive, Collection<String> negative, int top) {
        if (modelVectorsLoaded) {
            return word2Vec.wordsNearestSum(positive, negative, top);
        }
        throw new IllegalStateException(METHOD_NOT_AVAILABLE);
    }

    @Override
    public Map<String, Double> accuracy(List<String> questions) {
        if (modelVectorsLoaded) {
            return word2Vec.accuracy(questions);
        }
        throw new IllegalStateException(METHOD_NOT_AVAILABLE);
    }

    @Override
    public int indexOf(String word) {
        if (modelVectorsLoaded) {
            return word2Vec.indexOf(word);
        }
        return vocab().indexOf(word);
    }


    @Override
    public List<String> similarWordsInVocabTo(String word, double accuracy) {
        if (modelVectorsLoaded) {
            return word2Vec.similarWordsInVocabTo(word, accuracy);
        }
        throw new IllegalStateException(METHOD_NOT_AVAILABLE);
    }

    @Override
    public Collection<String> wordsNearest(Collection<String> positive, Collection<String> negative, int top) {
        if (modelVectorsLoaded) {
            return word2Vec.wordsNearest(positive, negative, top);
        }
        throw new IllegalStateException(METHOD_NOT_AVAILABLE);
    }


    @Override
    public Collection<String> wordsNearest(String word, int n) {
        if (modelVectorsLoaded) {
            return word2Vec.wordsNearest(word,n);
        }
        throw new IllegalStateException(METHOD_NOT_AVAILABLE);
    }


    @Override
    public double similarity(String word, String word2) {
        if (modelVectorsLoaded) {
            return word2Vec.similarity(word, word2);
        }
        throw new IllegalStateException(METHOD_NOT_AVAILABLE);
    }

    @Override
    public WeightLookupTable lookupTable() {
        if (modelVectorsLoaded) {
            return word2Vec.lookupTable();
        }
        return null;
    }

    @Override
    public void setModelUtils(ModelUtils utils) {
    }

    @Override
    public void loadWeightsInto(INDArray array) {}

    @Override
    public int vectorSize() {return -1;}

    @Override
    public boolean jsonSerializable() {return false;}

    public double getLearningRate() {
        return fastTextImpl.getLr();
    }

    public int getDimension() {
        return fastTextImpl.getDim();
    }

    public int getContextWindowSize() {
        return fastTextImpl.getContextWindowSize();
    }

    public int getEpoch() {
        return fastTextImpl.getEpoch();
    }

    public int getNegativesNumber() {
        return fastTextImpl.getNSampledNegatives();
    }

    public int getWordNgrams() {
        return fastTextImpl.getWordNgrams();
    }

    public String getLossName() {
        return fastTextImpl.getLossName();
    }

    public String getModelName() {
        return fastTextImpl.getModelName();
    }

    public int getNumberOfBuckets() {
        return fastTextImpl.getBucket();
    }

    public String getLabelPrefix() {
        return fastTextImpl.getLabelPrefix();
    }

    @Override
    public boolean outOfVocabularySupported() {
        return true;
    }

}
