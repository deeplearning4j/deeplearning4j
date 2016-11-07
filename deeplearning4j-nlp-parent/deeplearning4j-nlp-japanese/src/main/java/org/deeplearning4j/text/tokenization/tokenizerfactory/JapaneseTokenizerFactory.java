package org.deeplearning4j.text.tokenization.tokenizerfactory;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.JapaneseTokenizer;

import java.io.InputStream;

public class JapaneseTokenizerFactory implements TokenizerFactory {
  private TokenPreProcess preProcess;
  private boolean useBaseForm;

  public JapaneseTokenizerFactory() {
  }

  @Override
  public Tokenizer create(String toTokenize) {
    if (toTokenize.isEmpty()) {
      throw new IllegalArgumentException("Unable to proceed; no sentence to tokenize");
    }
    JapaneseTokenizer t = new JapaneseTokenizer(toTokenize);
    return t;
  }

  @Override
  public Tokenizer create(InputStream toTokenize) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void setTokenPreProcessor(TokenPreProcess preProcessor) {
    this.preProcess = preProcess;
  }

  @Override
  public TokenPreProcess getTokenPreProcessor() {
    return this.preProcess;
  }

}

