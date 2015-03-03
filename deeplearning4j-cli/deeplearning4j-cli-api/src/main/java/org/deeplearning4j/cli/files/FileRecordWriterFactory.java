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
import org.canova.api.records.reader.factory.RecordWriterFactory;
import org.canova.api.records.writer.RecordWriter;
import org.canova.api.records.writer.impl.CSVRecordWriter;
import org.canova.api.records.writer.impl.FileRecordWriter;
import org.canova.api.records.writer.impl.MatlabRecordWriter;
import org.canova.api.records.writer.impl.SVMLightRecordWriter;

import java.io.File;
import java.net.URI;

/**
 * Creates an instance of a file record writer
 *
 * @author sonali
 */
public class FileRecordWriterFactory implements RecordWriterFactory {
  /**
   * Creates new RecordWriter instance based on the output file extension
   *
   * @param uri destination for saving model
   * @return RecordWriter instance
   * @throws Exception
   */

  @Override
  public RecordWriter create(URI uri) throws Exception {
    Preconditions.checkArgument(uri != null, "URI needs to be specified");
    File out = new File(uri.toString());
    String fileNameExtension = FilenameUtils.getExtension(uri.toString()).toLowerCase();
    RecordWriter recordWriter;

    switch (fileNameExtension) {
      case "csv":
        recordWriter = new CSVRecordWriter(out);
        break;
      case "txt":
        recordWriter = new FileRecordWriter(out, true);
        break;
      case "mat":
        recordWriter = new MatlabRecordWriter(out);
        break;
      case "svmlight":
        recordWriter = new SVMLightRecordWriter(out, true);
        break;
      default:
        throw new UnknownFormatException("Unknown file format");
    }


    return recordWriter;
  }


}
