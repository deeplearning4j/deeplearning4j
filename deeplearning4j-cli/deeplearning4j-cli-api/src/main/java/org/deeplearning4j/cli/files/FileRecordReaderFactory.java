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

package org.deeplearning4j.cli.files;

import com.google.common.base.Preconditions;
import org.apache.commons.io.FilenameUtils;
import org.canova.api.exceptions.UnknownFormatException;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.factory.RecordReaderFactory;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.records.reader.impl.FileRecordReader;
import org.canova.api.records.reader.impl.MatlabRecordReader;
import org.canova.api.records.reader.impl.SVMLightRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;

import java.io.File;
import java.net.URI;

/**
 * Creates an instance of file record reader
 *
 * @author sonali
 */
public class FileRecordReaderFactory implements RecordReaderFactory {
  /**
   * Creates new RecordReader instance based on the input file extension
   *
   * @param uri location of input data
   * @return RecordReader instance
   * @throws UnknownFormatException
   */

  @Override
  public RecordReader create(URI uri) throws UnknownFormatException {
    Preconditions.checkArgument(uri != null, "URI cannot be null");
    File file = new File(uri.toString());
    InputSplit split = new FileSplit(file);

    String fileNameExtension = FilenameUtils.getExtension(uri.toString()).toLowerCase();
    RecordReader ret = null;
    switch (fileNameExtension) {
      case "csv":
        ret = new CSVRecordReader();
        break;
      case "txt":
        ret = new FileRecordReader();
        break;
      case "mat":
        ret = new MatlabRecordReader();
        break;
      case "svmlight":
        ret = new SVMLightRecordReader();
        break;
      default:
        throw new UnknownFormatException("Unknown file format");
    }

    try {
      ret.initialize(split);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    return ret;

  }


}
