package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.SparkConf;

import java.util.HashMap;
import java.util.Map;

/**
 * @author jeffreytang
 */
@Deprecated
public class Word2VecVariables {

    public final static String NAME_SPACE = "org.deeplearning4j.scaleout.perform.models.word2vec";
    public final static String VECTOR_LENGTH = NAME_SPACE + ".length";
    public final static String ADAGRAD = NAME_SPACE + ".adagrad";
    public final static String NEGATIVE = NAME_SPACE + ".negative";
    public final static String NUM_WORDS = NAME_SPACE + ".numwords";
    public final static String TABLE = NAME_SPACE + ".table";
    public final static String WINDOW = NAME_SPACE + ".window";
    public final static String ALPHA = NAME_SPACE + ".alpha";
    public final static String MIN_ALPHA = NAME_SPACE + ".minalpha";
    public final static String ITERATIONS = NAME_SPACE + ".iterations";
    public final static String N_GRAMS = NAME_SPACE + ".ngrams";
    public final static String TOKENIZER = NAME_SPACE + ".tokenizer";
    public final static String TOKEN_PREPROCESSOR = NAME_SPACE + ".preprocessor";
    public final static String REMOVE_STOPWORDS = NAME_SPACE + ".removestopwords";
    public final static String SEED = NAME_SPACE + ".SEED";

    public final static Map<String, Object> defaultVals = new HashMap<String, Object>() {
        {
            put(VECTOR_LENGTH, 100);
            put(ADAGRAD, false);
            put(NEGATIVE, 5);
            put(NUM_WORDS, 1);
            // TABLE would be a string of byte of the ndarray used for -ve sampling
            put(WINDOW, 5);
            put(ALPHA, 0.025);
            put(MIN_ALPHA, 1e-2);
            put(ITERATIONS, 1);
            put(N_GRAMS, 1);
            put(TOKENIZER, "org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory");
            put(TOKEN_PREPROCESSOR, "org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor");
            put(REMOVE_STOPWORDS, false);
            put(SEED, 42L);
        }
    };

    private Word2VecVariables() {}

    @SuppressWarnings("unchecked")
    public static <T> T getDefault(String variableName) {
        return (T) defaultVals.get(variableName);
    }

    @SuppressWarnings("unchecked")
    public static <T> T assignVar(String variableName, SparkConf conf, Class clazz) throws Exception {
        Object ret;
        if (clazz.equals(Integer.class)) {
            ret = conf.getInt(variableName, (Integer) getDefault(variableName));

        } else if (clazz.equals(Double.class)) {
            ret = conf.getDouble(variableName, (Double) getDefault(variableName));

        } else if (clazz.equals(Boolean.class)) {
            ret = conf.getBoolean(variableName, (Boolean) getDefault(variableName));

        } else if (clazz.equals(String.class)) {
            ret = conf.get(variableName, (String) getDefault(variableName));

        } else if (clazz.equals(Long.class)) {
            ret = conf.getLong(variableName, (Long) getDefault(variableName));
        } else {
            throw new Exception("Variable Type not supported. Only boolean, int, double and String supported.");
        }
        return (T) ret;
    }
}
