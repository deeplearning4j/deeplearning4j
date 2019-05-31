package org.deeplearning4j.models.fasttext;

import com.github.jfasttext.JFastText;
import lombok.Builder;
import lombok.Data;
import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.List;
import java.util.Map;

public class FastText implements WordVectors {

    private JFastText fastTextImpl;
    private Builder builder;

    private boolean modelLoaded;

    public FastText() {
        fastTextImpl = new JFastText();
    }

    public FastText(Builder builder) {
        this();
        this.builder = builder;
    }

    public void fit() {

        String[] cmd;
        if (builder.supervised)
            cmd = new String[]{"supervised", "-input", builder.inputFile,
                    "-output", builder.outputFile};
        else
            cmd = new String[]{"-input", builder.inputFile,
                    "-output", builder.outputFile};

        fastTextImpl.runCmd(cmd);
    }

    public void loadBinaryModel(String modelPath) {
        fastTextImpl.loadModel(modelPath);
        modelLoaded = true;
    }


    public String predict(String text) {

        if (!modelLoaded)
            throw new IllegalStateException("Model must be loaded before predict!");

        String label = fastTextImpl.predict(text);
        return label;
    }

    @Override
    public String getUNK() {
        return StringUtils.EMPTY;
    }

    @Override
    public void setUNK(String input) {

    }

    @Override
    public boolean hasWord(String word) {
        return false;
    }

    @Override
    public Collection<String> wordsNearest(INDArray words, int top) {
        return null;
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
    public double[] getWordVector(String word) {
        return null;
    }

    @Override
    public INDArray getWordVectorMatrixNormalized(String word) {
        return null;
    }

    @Override
    public INDArray getWordVectorMatrix(String word) {
        return null;
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
    public VocabCache vocab() {
        return null;
    }

    @Override
    public WeightLookupTable lookupTable() {
        return null;
    }

    @Override
    public void setModelUtils(ModelUtils utils) {

    }

    @Override
    public void loadWeightsInto(INDArray array) {}


    @Override
    public long vocabSize() {return -1;}

    @Override
    public int vectorSize() {return -1;}

    @Override
    public boolean jsonSerializable() {return false;}

    @Data
    @lombok.Builder
    public static class Builder {
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
    }
}
