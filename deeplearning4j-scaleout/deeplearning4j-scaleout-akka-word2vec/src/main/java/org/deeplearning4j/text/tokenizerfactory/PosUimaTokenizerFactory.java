package org.deeplearning4j.text.tokenizerfactory;

import java.util.Collection;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.cleartk.token.stem.snowball.DefaultSnowballStemmer;
import org.deeplearning4j.text.annotator.PoStagger;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.annotator.TokenizerAnnotator;
import org.deeplearning4j.text.tokenizer.PosUimaTokenizer;
import org.deeplearning4j.word2vec.tokenizer.Tokenizer;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;

/**
 * Creates a tokenizer that filters by 
 * part of speech tags
 * @see {org.deeplearning4j.text.tokenizer.PosUimaTokenizer}
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
            return AnalysisEngineFactory.createEngine(AnalysisEngineFactory.createEngineDescription(SentenceAnnotator.getDescription(), TokenizerAnnotator.getDescription(), PoStagger.getDescription("en"),DefaultSnowballStemmer.getDescription("English")));
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }


	@Override
	public Tokenizer create(String toTokenize) {
		return new PosUimaTokenizer(toTokenize,tokenizer,allowedPoSTags);
	}


}
