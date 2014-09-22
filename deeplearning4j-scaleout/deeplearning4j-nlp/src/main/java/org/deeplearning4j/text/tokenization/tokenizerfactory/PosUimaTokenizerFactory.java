package org.deeplearning4j.text.tokenization.tokenizerfactory;


import static  org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import static  org.apache.uima.fit.factory.AnalysisEngineFactory.createEngine;

import java.io.InputStream;
import java.util.Collection;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.deeplearning4j.text.annotator.PoStagger;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.annotator.StemmerAnnotator;
import org.deeplearning4j.text.annotator.TokenizerAnnotator;
import org.deeplearning4j.text.tokenization.tokenizer.PosUimaTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

/**
 * Creates a tokenizer that filters by 
 * part of speech tags
 * @see {org.deeplearning4j.text.tokenization.tokenizer.PosUimaTokenizer}
 * @author Adam Gibson
 *
 */
public class PosUimaTokenizerFactory implements TokenizerFactory {

	private AnalysisEngine tokenizer;
	private Collection<String> allowedPoSTags;


    public PosUimaTokenizerFactory(Collection<String> allowedPoSTags) {
             this(defaultAnalysisEngine(),allowedPoSTags);
    }

	public PosUimaTokenizerFactory(AnalysisEngine tokenizer,Collection<String> allowedPosTags) {
		this.tokenizer = tokenizer;
		this.allowedPoSTags = allowedPosTags;
	}


    public static AnalysisEngine defaultAnalysisEngine()  {
        try {
            return createEngine(
                    createEngineDescription(SentenceAnnotator.getDescription(),
                            TokenizerAnnotator.getDescription(),
                            PoStagger.getDescription("en"),
                            StemmerAnnotator.getDescription("English")));
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }


	@Override
	public Tokenizer create(String toTokenize) {
		return new PosUimaTokenizer(toTokenize,tokenizer,allowedPoSTags);
	}

	@Override
	public Tokenizer create(InputStream toTokenize) {
		// TODO Auto-generated method stub
		return null;
	}


}
