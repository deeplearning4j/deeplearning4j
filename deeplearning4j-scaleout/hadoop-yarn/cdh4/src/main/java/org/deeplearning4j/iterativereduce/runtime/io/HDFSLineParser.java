package org.deeplearning4j.iterativereduce.runtime.io;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.LineReader;
import org.deeplearning4j.iterativereduce.runtime.Updateable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;

public class HDFSLineParser<T extends Updateable> implements RecordParser<T> {

  private static final Logger LOG = LoggerFactory.getLogger(HDFSLineParser.class);

  Class<T> clazz;
  LinkedList<T> records;
  Configuration conf;
  FileSystem fs;
  Path file = null;
  long readOffset = 0;
  long readLength = 0;

  boolean parseComplete = false;
  int position = 0;

  public HDFSLineParser(Class<T> clazz) {
    this.clazz = clazz;
  }

  @Override
  public void reset() {
    position = 0;
  }

  /**
   * TODO: rebuild ----------------------------
   * -  stop loading entire split
   * 
   */
  @Override
  public void parse() {
    if (file == null)
      throw new IllegalStateException(
          "File cannot be null. Call setFile() before calling parse()");

    if (parseComplete)
      throw new IllegalStateException("File has alredy been parsed.");

    FSDataInputStream fis = null;
    BufferedReader br = null;

    try {
      fis = fs.open(file);
      br = new BufferedReader(new InputStreamReader(fis));

      // Go to our offset. DFS should take care of opening a block local file
      fis.seek(readOffset);
      records = new LinkedList<>();

      LineReader ln = new LineReader(fis);
      Text line = new Text();
      long read = readOffset;

      if (readOffset != 0)
        read += ln.readLine(line);

      while (read < readLength) {
        int r = ln.readLine(line);
        if (r == 0)
          break;

        try {
          T record = clazz.newInstance();
          record.fromString(line.toString());
          records.add(record);
        } catch (Exception ex) {
          LOG.warn("Unable to instantiate the updateable record type", ex);
        }
        read += r;
      }

    } catch (IOException ex) {
      LOG.error("Encountered an error while reading from file " + file, ex);
    } finally {
      try {
        if (br != null)
          br.close();

        if (fis != null)
          fis.close();
      } catch (IOException ex) {
        LOG.error("Can't close file", ex);
      }
    }

    LOG.debug("Read " + records.size() + " records");
  }

  @Override
  public void setFile(String f) {
    setFile(f, 0, Long.MAX_VALUE);
  }

  @Override
  public void setFile(String f, long start, long length) {
    if (conf == null)
      conf = new Configuration();

    readOffset = start;
    readLength = length;

    try {
      file = new Path(f);
      fs = file.getFileSystem(conf);

      // Don't read past the file length!
      FileStatus fstat = fs.getFileStatus(file);
      if (readLength > fstat.getLen())
        readLength = fstat.getLen();

      if (!fs.isFile(file)) {
        throw new IOException("File " + file
            + " is not a regular file, cannot read or parse");
      }
    } catch (IOException ex) {
      LOG.error("Unable to get file status for " + file, ex);
      file = null;
    }

    LOG.debug("Found a valid file, name=" + file.toString() + ", offset="
        + readOffset + ", length=" + readLength);
  }

  @Override
  public boolean hasMoreRecords() {
    if (records == null)
      return false;

    return (position < records.size());
  }

  @Override
  public T nextRecord() {
    if (records == null)
      return null;

    return records.get(position++);
  }

  @Override
  public int getCurrentRecordsProcessed() {
    // TODO Auto-generated method stub
    return 0;
  }
}
