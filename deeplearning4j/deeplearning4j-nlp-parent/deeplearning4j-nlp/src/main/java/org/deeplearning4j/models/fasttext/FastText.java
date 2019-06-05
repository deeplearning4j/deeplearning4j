package org.deeplearning4j.models.fasttext;

import com.github.jfasttext.JFastText;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.*;
import java.util.Collection;
import java.util.List;
import java.util.Map;

@Slf4j
@AllArgsConstructor
@lombok.Builder
public class FastText implements WordVectors, Serializable {

    @Getter private boolean supervised;
    @Getter private boolean quantize;
    @Getter private boolean predict;
    @Getter private boolean predict_prob;

    private boolean skipgram;
    @Builder.Default  private int bucket = 100;
    @Builder.Default  private int minCount = 1;

    @Getter private boolean cbow;
    @Getter private boolean nn;
    @Getter private boolean analogies;
    @Getter private String inputFile;
    @Getter private String outputFile;
    @Getter private SentenceIterator iterator;
    @Getter private String modelName;
    @Getter private String lossName;
    //TODO:
    @Getter private double[] pretrainedVectors;

    private transient JFastText fastTextImpl;
    @Getter private boolean modelLoaded;
    private VocabCache vocabCache;

    public FastText(File modelPath) {
        this();
        loadBinaryModel(modelPath.getAbsolutePath());
    }

    public FastText() {
        fastTextImpl = new JFastText();
    }

    public void init() {
        if (fastTextImpl == null)
            fastTextImpl = new JFastText();
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
        /*if (modelUtils == null) {
            val vocabCache = new AbstractCache.Builder().build();

            val lookupTable = new InMemoryLookupTable.Builder().cache(vocabCache).build();

            modelUtils = new BasicModelUtils<>();
            modelUtils.init(lookupTable);
        }*/
    }

    public void fit() {

        String[] cmd;
        if (skipgram) {
            cmd = new String[]{"skipgram", "-bucket", Integer.toString(bucket), "-minCount", Integer.toString(minCount),
                    "-input", inputFile, "-output", outputFile};
        }
        else if (cbow) {
            cmd = new String[]{"cbow", "-bucket", Integer.toString(bucket), "-minCount", Integer.toString(minCount),
                    "-input", inputFile, "-output", outputFile};
        }
        else if (supervised)
            cmd = new String[]{"supervised", "-input", inputFile,
                    "-output", outputFile};
        else
            cmd = new String[]{"-input", inputFile,
                    "-output", outputFile};
        fastTextImpl.runCmd(cmd);
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

    public String predict(String text) {

        if (!modelLoaded)
            throw new IllegalStateException("Model must be loaded before predict!");

        String label = fastTextImpl.predict(text);
        return label;
    }

    public Pair<String, Float> predictProbability(String text) {

        if (!modelLoaded)
            throw new IllegalStateException("Model must be loaded before predict!");

        JFastText.ProbLabel predictedProbLabel = fastTextImpl.predictProba(text);

        Pair<String,Float> retVal = new Pair<>();
        retVal.setFirst(predictedProbLabel.label);
        retVal.setSecond(predictedProbLabel.logProb);
        return retVal;
    }

    @Override
    public VocabCache vocab() {
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
        return vocabCache;
    }

    @Override
    public long vocabSize() {
        if (!modelLoaded)
            throw new IllegalStateException("Load model before calling vocab()");
        return fastTextImpl.getNWords();
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
        List<Float> vectors = fastTextImpl.getVector(word);
        double[] retVal = new double[vectors.size()];
        for (int i = 0; i < vectors.size(); ++i) {
            retVal[i] = vectors.get(i);
        }
        return retVal;
    }

    @Override
    public INDArray getWordVectorMatrixNormalized(String word) {
        INDArray r = getWordVectorMatrix(word);
        return r.divi(Nd4j.getBlasWrapper().nrm2(r));
    }

    @Override
    public INDArray getWordVectorMatrix(String word) {
        double[] values = getWordVector(word);
        return Nd4j.createFromArray(values);
    }

    @Override
    public INDArray getWordVectors(Collection<String> labels) {
        return null;
    }

    @Override
    public INDArray getWordVectorsMean(Collection<String> labels) {
        return null;
    }

    @Override
    public boolean hasWord(String word) {
        return fastTextImpl.getWords().contains(word);
    }

    protected transient ModelUtils modelUtils;

    @Override
    public Collection<String> wordsNearest(INDArray words, int top) {
        return modelUtils.wordsNearest(words, top);
    }

    @Override
    public Collection<String> wordsNearestSum(INDArray words, int top) {
        return modelUtils.wordsNearestSum(words, top);
    }

    @Override
    public Collection<String> wordsNearestSum(String word, int n) {
        return modelUtils.wordsNearestSum(word, n);
    }


    @Override
    public Collection<String> wordsNearestSum(Collection<String> positive, Collection<String> negative, int top) {
        return modelUtils.wordsNearestSum(positive, negative, top);
    }

    @Override
    public Map<String, Double> accuracy(List<String> questions) {
        return modelUtils.accuracy(questions);
    }

    @Override
    public int indexOf(String word) {
        return vocab().indexOf(word);
    }


    @Override
    public List<String> similarWordsInVocabTo(String word, double accuracy) {
        return modelUtils.similarWordsInVocabTo(word, accuracy);
    }

    @Override
    public Collection<String> wordsNearest(Collection<String> positive, Collection<String> negative, int top) {
        return modelUtils.wordsNearest(positive, negative, top);
    }


    @Override
    public Collection<String> wordsNearest(String word, int n) {
        return modelUtils.wordsNearestSum(word, n);
    }


    @Override
    public double similarity(String word, String word2) {
        return modelUtils.similarity(word, word2);
    }

    @Override
    public WeightLookupTable lookupTable() {
        return null;
    }

    @Override
    public void setModelUtils(ModelUtils utils) {
        this.modelUtils = utils;
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

}
