package org.deeplearning4j.text.tokenization.tokenizer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.cas.CAS;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.util.CasPool;
import org.cleartk.token.type.Token;
import org.deeplearning4j.text.uima.UimaResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tokenizer based on the passed in analysis engine
 * @author Adam Gibson
 *
 */
public class UimaTokenizer implements Tokenizer {

    private List<String> tokens;
    private int index;
    private static Logger log = LoggerFactory.getLogger(UimaTokenizer.class);
    private boolean checkForLabel;



    public UimaTokenizer(String tokens,UimaResource resource,boolean checkForLabel) {
    
        this.checkForLabel = checkForLabel;
        this.tokens = new ArrayList<>();
        try {
            CAS cas = resource.process(tokens);

            Collection<Token> tokenList = JCasUtil.select(cas.getJCas(), Token.class);

            for(Token t : tokenList) {

                if(!checkForLabel || valid(t.getCoveredText()))
                    if(t.getLemma() != null)
                        this.tokens.add(t.getLemma());
                    else if(t.getStem() != null)
                        this.tokens.add(t.getStem());
                    else
                        this.tokens.add(t.getCoveredText());
            }


           resource.release(cas);


        } catch (Exception e) {
        	e.printStackTrace();
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
        List<String> tokens = new ArrayList<>();
        while(hasMoreTokens()) {
            tokens.add(nextToken());
        }
        return tokens;
    }




}
