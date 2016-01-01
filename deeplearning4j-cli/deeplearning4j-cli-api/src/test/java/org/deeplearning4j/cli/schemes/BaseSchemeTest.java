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

package org.deeplearning4j.cli.schemes;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.writer.RecordWriter;
import org.canova.api.writable.Writable;
import org.deeplearning4j.cli.api.schemes.Scheme;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Tests scheme for creating record writers and readers
 * Initializes scheme
 * Creates a new record reader for a test data set
 * Asserts number of columns in reader
 * Uses a record writer to write to file
 * Creates another record reader to assert columns
 * and confirm that data was written to file
 *
 *  @author sonali
 */
public abstract class BaseSchemeTest {
  protected Scheme scheme;

  @Before
  public void before() {
    initScheme();
  }

  /**
   * Initializes the appropriate scheme for testing
   * @return scheme
   */
  public  Scheme getScheme() {
    return scheme;
  }

  public abstract void initScheme();

  @Test
  public void testScheme() throws Exception {
    Scheme scheme = getScheme();
    RecordReader recordReader = scheme.createReader(new ClassPathResource("iris.txt").getURI());

    assertTrue(recordReader.hasNext());
    while (recordReader.hasNext()) {
      Collection<Writable> record = recordReader.next();
      assertEquals(1, record.size());
    }

    recordReader = scheme.createReader(new ClassPathResource("iris.txt").getURI());

    List<Collection<Writable>> records = new ArrayList<>();
    int count = 0;
    RecordWriter writer = scheme.createWriter(URI.create("test_out.txt"));

    for (Collection<Writable> record : records) {
      writer.write(record);
      assertEquals(1, record.size());
    }

    recordReader = scheme.createReader(URI.create("test_out.txt"));

    while (recordReader.hasNext()) {
      Collection<Writable> record = recordReader.next();
      records.add(record);
      assertEquals(1, record.size());
      count++;
    }

    assertEquals(1, count);
    writer.close();
  }

}
