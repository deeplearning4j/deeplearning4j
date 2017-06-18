package org.deeplearning4j.text.tokenization.tokenizer;

import org.ansj.domain.Result;
import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.NlpAnalysis;
import org.ansj.splitWord.analysis.ToAnalysis;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;


/**
 * @date: June 2,2017
 * @author: wangfeng
 * @Description:The word of the open source segmentation algorithm is based on dictionaries
 *
 */

public class ChineseTokenizer implements Tokenizer{

    private TokenPreProcess tokenPreProcess;
    private List<Term> tokenList;
    private Iterator<Term> tokenIter;

    public ChineseTokenizer() {}
    public ChineseTokenizer(String toTokenize) {
        Result result = NlpAnalysis.parse(toTokenize);
        this.tokenList = result.getTerms();
        this.tokenIter = tokenList.iterator();
    }

    @Override
    public boolean hasMoreTokens() {
        return tokenIter.hasNext();
    }

    @Override
    public int countTokens() {
        return tokenList != null ? tokenList.size() : 0;
    }

    @Override
    public String nextToken() {
        if (!hasMoreTokens()) {
            throw new NoSuchElementException();
        }

        return this.tokenPreProcess != null ? this.tokenPreProcess.preProcess(tokenIter.next().getName()) : tokenIter.next().getName();
    }

    @Override
    public List<String> getTokens() {
        ArrayList tokenList = new ArrayList();

        while(hasMoreTokens()) {
            tokenList.add(nextToken());
        }
        return tokenList;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
        this.tokenPreProcess = tokenPreProcessor;
    }

}
