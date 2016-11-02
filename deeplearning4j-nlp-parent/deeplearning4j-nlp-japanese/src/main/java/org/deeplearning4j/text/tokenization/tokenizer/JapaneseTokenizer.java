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

import com.atilika.kuromoji.TokenizerBase;
import com.atilika.kuromoji.ipadic.Token;
import com.atilika.kuromoji.ipadic.Tokenizer;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * modified by kepricon on 16. 10. 28.
 * A thin wrapper for Japanese Morphological Analyzer Kuromoji (ver.0.9.0),
 * it tokenizes texts which is written in languages
 * that words are not separated by whitespaces.
 *
 * In thenory, Kuromoji is a language-independent Morphological Analyzer library,
 * so if you want to tokenize non-Japanese texts (Chinese, Korean etc.),
 * you can do it with MeCab style dictionary for each languages.
 */
public class JapaneseTokenizer implements org.deeplearning4j.text.tokenization.tokenizer.Tokenizer {

  private Iterator<String> tokenIter;
  private List<String> tokens;

  private TokenPreProcess preProcess;

  public JapaneseTokenizer(String toTokenize) {
    Tokenizer tokenizer = new Tokenizer();
    Iterator<Token> iter = tokenizer.tokenize(toTokenize).iterator();

    tokens = new ArrayList<String>();

    while(iter.hasNext()){
//      tokens.add(iter.next().getBaseForm());
      tokens.add(iter.next().getSurface());
    }

    tokenIter = this.tokens.iterator();
  }

  @Override
  public boolean hasMoreTokens() {
    return tokenIter.hasNext();
  }

  @Override
  public int countTokens() {
    return tokens.size();
  }

  @Override
  public String nextToken() {
    if(hasMoreTokens() == false){
      throw new NoSuchElementException();
    }
    return this.preProcess != null ? this.preProcess.preProcess(tokenIter.next()) : tokenIter.next();
  }

  @Override
  public List<String> getTokens() {
    return tokens;
  }

  @Override
  public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
    this.preProcess = tokenPreProcessor;
  }

}

