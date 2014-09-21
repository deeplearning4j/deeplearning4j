package org.deeplearning4j.text.tokenization.tokenizerfactory;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.CasPool;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.annotator.StemmerAnnotator;
import org.deeplearning4j.text.annotator.TokenizerAnnotator;
import org.deeplearning4j.text.tokenization.tokenizer.UimaTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;


/**
 * Uses a uima {@link AnalysisEngine} to 
 * tokenize text.
 *
 *
 * @author Adam Gibson
 *
 */
public class UimaTokenizerFactory implements TokenizerFactory {

    private AnalysisEngine tokenizer;
    private CasPool pool;
    private boolean checkForLabel;
    private static AnalysisEngine defaultAnalysisEngine;


    public UimaTokenizerFactory() throws ResourceInitializationException {
        this(defaultAnalysisEngine(),true);
    }



    public UimaTokenizerFactory(AnalysisEngine tokenizer) {
        this(tokenizer,true);
    }


    public UimaTokenizerFactory(boolean checkForLabel) throws ResourceInitializationException {
        this(defaultAnalysisEngine(),checkForLabel);
    }



    public UimaTokenizerFactory(AnalysisEngine tokenizer,boolean checkForLabel) {
        super();
        this.tokenizer = tokenizer;
        this.checkForLabel = checkForLabel;
        try {
            pool = new CasPool(Runtime.getRuntime().availableProcessors() * 10,tokenizer);

        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }



    @Override
    public  Tokenizer create(String toTokenize) {
        if(tokenizer == null || pool == null)
            throw new IllegalStateException("Unable to proceed; tokenizer or pool is null");
        if(toTokenize == null || toTokenize.isEmpty())
            throw new IllegalArgumentException("Unable to proceed; on sentence to tokenize");
        return new UimaTokenizer(toTokenize,tokenizer,pool,checkForLabel);
    }


    /**
     * Creates a tokenization,/stemming pipeline
     * @return a tokenization/stemming pipeline
     */
    public static AnalysisEngine defaultAnalysisEngine()  {
        try {
            if(defaultAnalysisEngine == null)

                defaultAnalysisEngine =  AnalysisEngineFactory.createEngine(AnalysisEngineFactory.createEngineDescription(
                        SentenceAnnotator.getDescription(),
                        TokenizerAnnotator.getDescription(),
                        StemmerAnnotator.getDescription("English")));

            return defaultAnalysisEngine;
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }


}
