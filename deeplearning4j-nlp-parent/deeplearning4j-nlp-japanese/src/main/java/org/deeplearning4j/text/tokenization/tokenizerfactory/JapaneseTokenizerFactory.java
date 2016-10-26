package org.deeplearning4j.text.tokenization.tokenizerfactory;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.JapaneseTokenizer;

import java.io.InputStream;

public class JapaneseTokenizerFactory implements TokenizerFactory {
  private org.atilika.kuromoji.Tokenizer tokenizer;
  private TokenPreProcess preProcess;
  private boolean useBaseForm;

  public JapaneseTokenizerFactory() {
    this(
            org.atilika.kuromoji.Tokenizer.builder().mode(org.atilika.kuromoji.Tokenizer.Mode.NORMAL).build(),
            false
    );
  }

  public JapaneseTokenizerFactory(org.atilika.kuromoji.Tokenizer.Mode mode, boolean useBaseForm) {
    this(
            org.atilika.kuromoji.Tokenizer.builder().mode(mode).build(),
            useBaseForm
    );
  }

  // If you want further customization, you can give a raw kuromoji's tokenizer.
  public JapaneseTokenizerFactory(org.atilika.kuromoji.Tokenizer tokenizer, boolean useBaseForm) {
    this.tokenizer = tokenizer;
    this.useBaseForm = useBaseForm;
  }

  @Override
  public Tokenizer create(String toTokenize) {
    if (toTokenize == null || toTokenize.isEmpty()) {
      throw new IllegalArgumentException("Unable to proceed; no sentence to tokenize");
    }
    Tokenizer ret = new JapaneseTokenizer(tokenizer, toTokenize, useBaseForm);
    return ret;
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

