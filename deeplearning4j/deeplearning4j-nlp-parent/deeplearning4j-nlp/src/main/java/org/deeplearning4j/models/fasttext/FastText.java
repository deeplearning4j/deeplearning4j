package org.deeplearning4j.models.fasttext;

import com.github.jfasttext.JFastText;
import lombok.*;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;

@AllArgsConstructor
@lombok.Builder
public class FastText implements WordVectors {

    private boolean supervised;
    private boolean quantize;
    private boolean predict;
    private boolean predict_prob;
    private boolean skipgram;
    private boolean cbow;
    private boolean nn;
    private boolean analogies;
    private String inputFile;
    private String outputFile;

    private JFastText fastTextImpl;
    private boolean modelLoaded;

    public FastText(File modelPath) {
        this();
        loadBinaryModel(modelPath.getAbsolutePath());
    }

    public FastText(SentenceIterator iterator) {
        try {
            File tempFile = File.createTempFile("FTX", ".txt");
            BufferedWriter writer = new BufferedWriter(new FileWriter(tempFile));
            while (iterator.hasNext()) {
                String sentence = iterator.nextSentence();
                writer.write(sentence);
            }

            fastTextImpl = new JFastText();
        } catch (IOException e) {
            System.out.println(e.toString());
        }
    }

    public FastText() {
        fastTextImpl = new JFastText();
    }

    public void init() {
        fastTextImpl = new JFastText();
    }

    public void fit() {

        String[] cmd;
        if (supervised)
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

    public String predict(String text) {

        if (!modelLoaded)
            throw new IllegalStateException("Model must be loaded before predict!");

        String label = fastTextImpl.predict(text);
        return label;
    }

    private VocabCache vocabCache;

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
        return r.div(Nd4j.getBlasWrapper().nrm2(r));
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

    protected transient ModelUtils modelUtils = new BasicModelUtils<>();

    @Override
    public Collection<String> wordsNearest(INDArray words, int top) {
        return modelUtils.wordsNearest(words, top);
    }

    @Override
    public Collection<String> wordsNearestSum(INDArray words, int top) {
        return null;
    }

    @Override
    public Collection<String> wordsNearestSum(String word, int n) {
        return null;
    }


    @Override
    public Collection<String> wordsNearestSum(Collection<String> positive, Collection<String> negative, int top) {
        return null;
    }

    @Override
    public Map<String, Double> accuracy(List<String> questions) {
        return null;
    }

    @Override
    public int indexOf(String word) {return -1;}


    @Override
    public List<String> similarWordsInVocabTo(String word, double accuracy) {
        return null;
    }

    @Override
    public Collection<String> wordsNearest(Collection<String> positive, Collection<String> negative, int top) {
        return null;
    }


    @Override
    public Collection<String> wordsNearest(String word, int n) {
        return null;
    }


    @Override
    public double similarity(String word, String word2) {
        return -1.0;
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
}
