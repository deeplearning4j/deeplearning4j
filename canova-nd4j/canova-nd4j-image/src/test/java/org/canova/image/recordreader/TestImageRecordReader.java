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

package org.canova.image.recordreader;

import java.util.Collection;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.FileRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputStreamInputSplit;
import org.canova.api.writable.Writable;
import org.junit.Test;
import org.canova.api.util.ClassPathResource;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;


/**
 * @author Adam Gibson
 */
public class TestImageRecordReader {

    // TODO fix tests and fix for TravisCI
//    @Test
//    public void testInputStream() throws Exception {
//        RecordReader reader = new ImageRecordReader(28,28,false);
//        // keeps needlessly blowing up
//        ClassPathResource res = new ClassPathResource("/test.jpg");
//        reader.initialize(new InputStreamInputSplit(res.getInputStream(), res.getURI()));
//        assertTrue(reader.hasNext());
//        Collection<Writable> record = reader.next();
//        assertEquals(784,record.size());
//
//    }
//
//    @Test
//    public void testMultipleChannels() throws Exception {
//        RecordReader reader = new ImageRecordReader(28,28,3,false);
//        // keeps needlessly blowing up
//        ClassPathResource res = new ClassPathResource("/test.jpg");
//        reader.initialize(new InputStreamInputSplit(res.getInputStream(), res.getURI()));
//        assertTrue(reader.hasNext());
//        Collection<Writable> record = reader.next();
//        assertEquals(784 * 3,record.size());
//    }
//
//    @Test
//    public void testGetLabel() throws Exception {
//        RecordReader reader = new ImageNameRecordReader(28,28,3,true);
//        // keeps needlessly blowing up
//        ClassPathResource res = new ClassPathResource("/test-1.jpg");
//        reader.initialize(new InputStreamInputSplit(res.getInputStream(), res.getURI()));
//        assertTrue(reader.hasNext());
//        Collection<Writable> record = reader.next();
//        assertEquals(784 * 3 + 1, record.size());
//    }
//


}
