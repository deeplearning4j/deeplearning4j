/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.text.tokenization.tokenizer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import lombok.NonNull;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.cas.CAS;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.cleartk.token.type.Sentence;
import org.cleartk.token.type.Token;
import org.deeplearning4j.text.annotator.PoStagger;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.annotator.StemmerAnnotator;
import org.deeplearning4j.text.annotator.TokenizerAnnotator;

/**
 * Filter by part of speech tag.
 * Any not valid part of speech tags
 * become NONE
 * @author Adam Gibson
 *
 */
public class PosUimaTokenizer  implements Tokenizer {

    private static AnalysisEngine engine;
    private List<String> tokens;
    private Collection<String> allowedPosTags;
    private int index;
    private static CAS cas;
    private TokenPreProcess preProcessor;
    private boolean stripNones = false;

    public PosUimaTokenizer(String tokens,AnalysisEngine engine,Collection<String> allowedPosTags) {
        this(tokens, engine, allowedPosTags, false);
    }

    public PosUimaTokenizer(String tokens,AnalysisEngine engine,Collection<String> allowedPosTags, boolean stripNones) {
        if(PosUimaTokenizer.engine == null)
            PosUimaTokenizer.engine = engine;
        this.allowedPosTags = allowedPosTags;
        this.tokens = new ArrayList<String>();
        this.stripNones = stripNones;
        try {
            if(cas == null)
                cas = engine.newCAS();

            cas.reset();
            cas.setDocumentText(tokens);
            PosUimaTokenizer.engine.process(cas);
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
        String ret = tokens.get(index); // preProcessor != null ? preProcessor.preProcess(tokens.get(index)) : tokens.get(index);
        index++;
        return ret;
    }

    @Override
    public List<String> getTokens() {
        List<String> tokens = new ArrayList<String>();
        while(hasMoreTokens()) {
            String nextT = nextToken();
            if (stripNones && nextT.equals("NONE"))
                continue;
            tokens.add(preProcessor != null ? preProcessor.preProcess(nextT) : nextT);
        }
        return tokens;
    }

    public static AnalysisEngine defaultAnalysisEngine()  {
        try {
            return AnalysisEngineFactory.createEngine(AnalysisEngineFactory.createEngineDescription(SentenceAnnotator.getDescription(), TokenizerAnnotator.getDescription(), PoStagger.getDescription("en"), StemmerAnnotator.getDescription("English")));
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }

	@Override
	public void setTokenPreProcessor(@NonNull TokenPreProcess tokenPreProcessor) {
		this.preProcessor = tokenPreProcessor;
	}




}
