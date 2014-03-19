package org.deeplearning4j.text.tokenizer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.cas.CAS;
import org.cleartk.token.type.Sentence;
import org.cleartk.token.type.Token;
import org.deeplearning4j.word2vec.tokenizer.Tokenizer;
import org.uimafit.util.JCasUtil;

/**
 * Filter by part of speech tag.
 * Any not valid part of speech tags
 * become NONE
 * @author Adam Gibson
 *
 */
public class PosUimaTokenizer  implements Tokenizer {

	private AnalysisEngine engine;
	private List<String> tokens;
	private Collection<String> allowedPosTags;
	private int index;

	public PosUimaTokenizer(String tokens,AnalysisEngine engine,Collection<String> allowedPosTags) {
		this.engine = engine;
		this.allowedPosTags = allowedPosTags;
		this.tokens = new ArrayList<String>();
		try {
			CAS cas = this.engine.newCAS();
			cas.setDocumentText(tokens);
			synchronized(engine) {
				this.engine.process(cas);
				for(Sentence s : JCasUtil.select(cas.getJCas(), Sentence.class)) {
					for(Token t : JCasUtil.selectCovered(Token.class,s)) {
						//add NONE for each invalid token
						if(valid(t))
							if(t.getLemma() != null)
								this.tokens.add(t.getLemma());
							else if(t.getStem() != null)
								this.tokens.add(t.getStem());
							else
								this.tokens.add(t.getCoveredText());
						else
							this.tokens.add("NONE");
					}
				}

			}


		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}

	private boolean valid(Token token) {
		String check = token.getCoveredText();
		if(check.matches("<[A-Z]+>") || check.matches("</[A-Z]+>"))
			return false;
		else if(token.getPos() != null && !this.allowedPosTags.contains(token.getPos()))
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
