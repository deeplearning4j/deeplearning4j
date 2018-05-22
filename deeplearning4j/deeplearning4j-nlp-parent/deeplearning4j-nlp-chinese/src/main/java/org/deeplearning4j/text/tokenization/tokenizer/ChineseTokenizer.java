package org.deeplearning4j.text.tokenization.tokenizer;

import org.ansj.domain.Result;
import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.NlpAnalysis;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;


/**
 * The ansj_seg of the open source segmentation algorithm comes form github,the link: https://github.com/NLPchina/ansj_seg
 * When the open source code that obeyed the Apache 2.0 license is reused, its latest commit ID is dedc45fdf85dfd2d4c691fb1f147d7cbf9a5d7fb
 * and  its copyright 2011-2016

 * @author: wangfeng
 * @since : June 2,2017
 */

public class ChineseTokenizer implements Tokenizer {

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
        return this.tokenPreProcess != null ? this.tokenPreProcess.preProcess(tokenIter.next().getName())
                        : tokenIter.next().getName();
    }

    @Override
    public List<String> getTokens() {
        ArrayList tokenList = new ArrayList();

        while (hasMoreTokens()) {
            tokenList.add(nextToken());
        }
        return tokenList;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
        this.tokenPreProcess = tokenPreProcessor;
    }

}
