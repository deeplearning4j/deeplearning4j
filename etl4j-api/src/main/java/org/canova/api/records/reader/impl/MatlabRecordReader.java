/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.api.records.reader.impl;


import org.canova.api.io.data.DoubleWritable;
import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;

import java.io.*;
import java.net.URI;
import java.util.*;

/**
 * Matlab record reader
 *
 * @author Adam Gibson
 */
public class MatlabRecordReader extends FileRecordReader {

  private List<Collection<Writable>> records = new ArrayList<>();
  private Iterator<Collection<Writable>> currIter;

  @Override
  public boolean hasNext() {
    return super.hasNext();
  }

  @Override
  public Collection<Writable> next() {
    //use the current iterator
    if (currIter != null && currIter.hasNext())
      return currIter.next();
    records.clear();
    //next file
    Collection<Writable> next = super.next();
    String val = next.iterator().next().toString();
    StringReader reader = new StringReader(val);
    int c;
    char chr;
    StringBuilder fileContent;
    boolean isComment;


    Collection<Writable> currRecord = new ArrayList<>();
    fileContent = new StringBuilder();
    isComment = false;
    records.add(currRecord);
    try {
      // determine number of attributes
      while ((c = reader.read()) != -1) {
        chr = (char) c;

        // comment found?
        if (chr == '%')
          isComment = true;

        // end of line reached
        if ((chr == '\n') || (chr == '\r')) {
          isComment = false;
          if (fileContent.length() > 0)
            currRecord.add(new DoubleWritable(new Double(fileContent.toString())));

          if (currRecord.size() > 0) {
            currRecord = new ArrayList<>();
            records.add(currRecord);
          }
          fileContent = new StringBuilder();
          continue;
        }

        // skip till end of comment line
        if (isComment)
          continue;

        // separator found?
        if ((chr == '\t') || (chr == ' ')) {
          if (fileContent.length() > 0) {
            currRecord.add(new DoubleWritable(new Double(fileContent.toString())));
            fileContent = new StringBuilder();
          }
        } else {
          fileContent.append(chr);
        }
      }

      // last number?
      if (fileContent.length() > 0)
        currRecord.add(new DoubleWritable(new Double(fileContent.toString())));


      currIter = records.iterator();

    } catch (Exception ex) {
      ex.printStackTrace();
      throw new IllegalStateException("Unable to determine structure as Matlab ASCII file: " + ex);
    }
    throw new IllegalStateException("Strange state detected");
  }

  @Override
  public Collection<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
    throw new UnsupportedOperationException("Reading Matlab data from DataInputStream: not yet implemented");
  }
}
