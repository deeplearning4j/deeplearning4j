package org.canova.cli.vectorization;

import java.io.IOException;
import java.util.Collection;

import org.canova.api.writable.Writable;
import org.canova.cli.shuffle.Shuffler;
import org.canova.cli.transforms.text.nlp.TfidfTextVectorizerTransform;

public class TextVectorizationEngine extends VectorizationEngine {

  /**
   * Currently the stock input format / RR gives us a vector already converted
   * -	TODO: separate this into a transform plugin
   * <p/>
   * <p/>
   * Thoughts
   * -	Inside the vectorization engine is a great place to put a pluggable transformation system [ TODO: v2 ]
   * -	example: MNIST binarization could be a pluggable transform
   * -	example: custom thresholding on blocks of pixels
   * <p/>
   * <p/>
   * Text Pipeline specific stuff
   * -	so right now the TF-IDF stuff has 2 major issues
   * 1.	its not parallelizable in its current form (loading words into memory doesnt scale)
   * 2.	vectorization is embedded in the inputformat/recordreader - which is conflating functionality
   */
  @Override
  public void execute() throws IOException {


    //	System.out.println( "TextVectorizationEngine > execute() [ START ]" );

    TfidfTextVectorizerTransform tfidfTransform = new TfidfTextVectorizerTransform();
    conf.setInt(TfidfTextVectorizerTransform.MIN_WORD_FREQUENCY, 1);
    //	conf.set(TfidfTextVectorizerTransform.TOKENIZER, "org.canova.nlp.tokenization.tokenizerfactory.PosUimaTokenizerFactory");
    tfidfTransform.initialize(conf);

    int recordsSeen = 0;


    // 1. collect stats for normalize
    while (reader.hasNext()) {

      // get the record from the input format
      Collection<Writable> w = reader.next();
      tfidfTransform.collectStatistics(w);
      recordsSeen++;

    }

    if (this.printStats) {

      System.out.println("Total Records: " + recordsSeen);
      System.out.println("Total Labels: " + tfidfTransform.getNumberOfLabelsSeen());
      System.out.println("Vocabulary Size of Corpus: " + tfidfTransform.getVocabularySize());
      tfidfTransform.debugPrintVocabList();

    }

    // 2. reset reader

    reader.close();
    //RecordReader reader = null;
    try {
      this.reader = inputFormat.createReader(split, conf);
    } catch (InterruptedException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }

    // 3. transform data

    if (shuffleOn) {

      Shuffler shuffle = new Shuffler();


      // collect the writables into the shuffler

      while (reader.hasNext()) {

        // get the record from the input format
        Collection<Writable> w = reader.next();
        tfidfTransform.transform(w);

        shuffle.addRecord(w);

      }


      // now send the shuffled data out

      while (shuffle.hasNext()) {

        Collection<Writable> shuffledRecord = shuffle.next();
        writer.write(shuffledRecord);

      }

      reader.close();
      writer.close();


    } else {

      while (reader.hasNext()) {

        // get the record from the input format
        Collection<Writable> w = reader.next();
        tfidfTransform.transform(w);

        // the reader did the work for us here
        writer.write(w);

      }


      reader.close();
      writer.close();

    }


  }


}
