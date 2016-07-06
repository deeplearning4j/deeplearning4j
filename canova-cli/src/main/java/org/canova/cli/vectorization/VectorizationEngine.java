package org.canova.cli.vectorization;

import java.io.IOException;
import java.util.Properties;

import org.canova.api.conf.Configuration;
import org.canova.api.exceptions.CanovaException;
import org.canova.api.formats.input.InputFormat;
import org.canova.api.formats.output.OutputFormat;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.writer.RecordWriter;
import org.canova.api.split.InputSplit;
import org.canova.cli.subcommands.Vectorize;

public abstract class VectorizationEngine {

  protected InputFormat inputFormat = null;
  protected OutputFormat outputFormat = null;
  protected InputSplit split = null;
  protected RecordWriter writer = null;
  protected RecordReader reader = null;
  protected Properties configProps = null;
  protected String outputFilename = null;
  protected Configuration conf = null;
  protected boolean shuffleOn = false;
  protected boolean normalizeData = true;
  protected boolean printStats = false;

  public void initialize(InputSplit split, InputFormat inputFormat, OutputFormat outputFormat, RecordReader reader, RecordWriter writer, Properties configProps, String outputFilename, Configuration conf) {

    this.split = split;
    this.reader = reader;
    this.writer = writer;
    this.configProps = configProps;
    this.inputFormat = inputFormat;
    this.outputFormat = outputFormat;
    this.outputFilename = outputFilename;
    this.conf = conf;


    if (null != this.configProps.get(Vectorize.SHUFFLE_DATA_FLAG)) {

      String shuffleValue = (String) this.configProps.get(Vectorize.SHUFFLE_DATA_FLAG);
      if ("true".equals(shuffleValue)) {
        shuffleOn = true;
      }

      System.out.println("Shuffle was turned on for this dataset.");
    }


    if (null != this.configProps.get(Vectorize.NORMALIZE_DATA_FLAG)) {
      String normalizeValue = (String) this.configProps.get(Vectorize.NORMALIZE_DATA_FLAG);
      if ("false".equals(normalizeValue)) {
        normalizeData = false;
      }

      System.out.println("Normalization was turned off for this dataset.");
    }

    if (null != this.configProps.get(Vectorize.PRINT_STATS_FLAG)) {
      String printSchema = (String) this.configProps.get(Vectorize.PRINT_STATS_FLAG);
      if ("true".equals(printSchema.trim().toLowerCase())) {
        //this.debugLoadedConfProperties();
        //this.inputSchema.debugPringDatasetStatistics();
        this.printStats = true;
      }
    }


  }

  public abstract void execute() throws CanovaException, IOException, InterruptedException;

  /**
   * These two methods are stubbing the future vector transform transform system
   * <p/>
   * We want to separate the transform logic from the inputformat/recordreader
   * -	example: a "thresholding" function that binarizes the vector entries
   * -	example: a sampling function that takes a larger images and down-samples the image into a small vector
   */
  public void addTransform() {
    throw new UnsupportedOperationException();
  }

  public void applyTransforms() {
    throw new UnsupportedOperationException();
  }


}
