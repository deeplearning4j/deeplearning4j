package org.deeplearning4j.text.tokenizer;

import java.util.ArrayList;
import java.util.List;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.cas.CAS;
import org.cleartk.token.type.Sentence;
import org.cleartk.token.type.Token;
import org.deeplearning4j.word2vec.tokenizer.Tokenizer;
import org.uimafit.util.JCasUtil;

/**
 * Tokenizer based on the passed in analysis engine
 * @author Adam Gibson
 *
 */
public class UimaTokenizer implements Tokenizer {

	private AnalysisEngine engine;
	private List<String> tokens;
	private int index;

	public UimaTokenizer(String tokens,AnalysisEngine engine) {
		this.engine = engine;
		
		this.tokens = new ArrayList<String>();
		try {
			CAS cas = this.engine.newCAS();
			cas.setDocumentText(tokens);
			synchronized(engine) {
				this.engine.process(cas);
				for(Sentence s : JCasUtil.select(cas.getJCas(), Sentence.class)) {
					for(Token t : JCasUtil.selectCovered(Token.class,s)) {
						if(valid(t.getCoveredText()))
							if(t.getLemma() != null)
								this.tokens.add(t.getLemma());
							else if(t.getStem() != null)
								this.tokens.add(t.getStem());
							else
								this.tokens.add(t.getCoveredText());
					}
				}

			}


		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}

	private boolean valid(String check) {
		if(check.matches("<[A-Z]+>") || check.matches("</[A-Z]+>"))
			return false;
		return true;
	}



	@Override
	public boolean hasMoreTokens() {
		return index < tokens.size();
	}

	@Override
	public int countTokens() {
		return tokens.size();
	}

	@Override
	public String nextToken() {
		String ret = tokens.get(index);
		index++;
		return ret;
	}

	@Override
	public List<String> getTokens() {
		List<String> tokens = new ArrayList<String>();
		while(hasMoreTokens()) {
			tokens.add(nextToken());
		}
		return tokens;
	}




}
