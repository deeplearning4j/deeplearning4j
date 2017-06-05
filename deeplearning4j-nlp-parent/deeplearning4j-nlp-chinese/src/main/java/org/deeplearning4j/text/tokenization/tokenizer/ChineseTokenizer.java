package org.deeplearning4j.text.tokenization.tokenizer;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import org.apdplat.word.WordSegmenter;
import org.apdplat.word.segmentation.Word;

/**
 * @date: June 2,2017
 * @author: wangfeng
 * @Description:The word of the open source segmentation algorithm is based on dictionaries
 *
 */

public class ChineseTokenizer implements Tokenizer{

    private TokenPreProcess tokenPreProcess;
    private List<Word> tokenList;
    private Iterator<Word> tokenIter;

    public ChineseTokenizer() {}
    public ChineseTokenizer(String toTokenize) {
        this.tokenList = WordSegmenter.seg(toTokenize);;
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
        return this.tokenPreProcess != null ? this.tokenPreProcess.preProcess(tokenIter.next().toString()) : tokenIter.next().toString();
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
