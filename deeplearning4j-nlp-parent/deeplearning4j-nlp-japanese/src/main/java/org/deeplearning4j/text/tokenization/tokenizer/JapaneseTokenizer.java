package org.deeplearning4j.text.tokenization.tokenizer;

import org.atilika.kuromoji.Token;
import org.atilika.kuromoji.Tokenizer;
import org.atilika.kuromoji.Tokenizer.Mode;

import java.util.ArrayList;
import java.util.List;

//  A thin wrapper for Japanese Morphological Analyzer Kuromoji (ver.0.7.7),
// it tokenizes texts which is written in languages
// that words are not separated by whitespaces.
//
// In thenory, Kuromoji is a language-independent Morphological Analyzer library,
// so if you want to tokenize non-Japanese texts (Chinese, Korean etc.),
// you can do it with MeCab style dictionary for each languages.
public class JapaneseTokenizer implements org.deeplearning4j.text.tokenization.tokenizer.Tokenizer {

  private List<String> tokens;
  private List<String> originalTokens;
  private int index;
  private org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess preProcess;
  private Tokenizer tokenizer;

  public JapaneseTokenizer(String toTokenize) {
    this(toTokenize, Mode.NORMAL, false);
  }

  // You can choose Segmentation Mode from options
  // Mode.NORMAL - recommend
  // Mode.SEARCH
  // Mode.EXTENDED
  public JapaneseTokenizer(String toTokenize, Mode mode, boolean useBaseForm) {
    this(
            org.atilika.kuromoji.Tokenizer.builder().mode(mode).build(),
            toTokenize,
            useBaseForm
    );
  }

  // This is used by JapaneseTokenizerFactory
  public JapaneseTokenizer(Tokenizer tokenizer, String toTokenize, boolean useBaseForm) {
    this.tokens = new ArrayList<>();
    this.tokenizer = tokenizer;

    for (Token token : tokenizer.tokenize(toTokenize)) {
      if (useBaseForm) {
        tokens.add(token.getBaseForm());
      } else {
        tokens.add(token.getSurfaceForm());
      }
    }

    index = tokens.size() > 0 ? 0 : -1;
  }
  @Override
  public boolean hasMoreTokens() {
    if (index < 0) {
      return false;
    } else {
      return index < tokens.size();
    }
  }

  @Override
  public int countTokens() {
    return tokens.size();
  }

  @Override
  public String nextToken() {
    if (index < 0) {
      return null;
    }

    String ret = tokens.get(index);
    index++;
    return preProcess != null ? preProcess.preProcess(ret) : ret;
  }

  @Override
  public List<String> getTokens() {
    List<String> tokens = new ArrayList<>();
    while (hasMoreTokens()) {
      tokens.add(nextToken());
    }
    return tokens;
  }

  @Override
  public void setTokenPreProcessor(org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess tokenPreProcessor) {
    this.preProcess = tokenPreProcessor;
  }

  public void resetIterator() {
    index = countTokens() > 0 ? 0 : -1;
  }
}

