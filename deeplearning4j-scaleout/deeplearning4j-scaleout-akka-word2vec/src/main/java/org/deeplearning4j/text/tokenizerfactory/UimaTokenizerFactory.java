package org.deeplearning4j.text.tokenizerfactory;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.token.stem.snowball.DefaultSnowballStemmer;
import org.deeplearning4j.text.annotator.PoStagger;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.annotator.TokenizerAnnotator;
import org.deeplearning4j.text.tokenizer.UimaTokenizer;
import org.deeplearning4j.word2vec.tokenizer.Tokenizer;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;


/**
 * Uses a uima {@link AnalysisEngine} to 
 * tokenize text.
 *
 *
 * @author agibsonccc
 *
 */
public class UimaTokenizerFactory implements TokenizerFactory {

    private AnalysisEngine tokenizer;

    public UimaTokenizerFactory() throws ResourceInitializationException {
        this(defaultAnalysisEngine());
    }


    public UimaTokenizerFactory(AnalysisEngine tokenizer) {
        super();
        this.tokenizer = tokenizer;
    }



    @Override
    public Tokenizer create(String toTokenize) {
        return new UimaTokenizer(toTokenize,tokenizer);
    }


    public static AnalysisEngine defaultAnalysisEngine()  {
        try {
            return AnalysisEngineFactory.createEngine(AnalysisEngineFactory.createEngineDescription(SentenceAnnotator.getDescription(), TokenizerAnnotator.getDescription(), DefaultSnowballStemmer.getDescription("English")));
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }


}
