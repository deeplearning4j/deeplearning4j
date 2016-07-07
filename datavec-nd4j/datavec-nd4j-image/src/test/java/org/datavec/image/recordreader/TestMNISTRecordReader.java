/*
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.image.recordreader;

import static org.junit.Assert.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Collection;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.writer.impl.misc.SVMLightRecordWriter;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.writable.ArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.mnist.MnistFetcher;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestMNISTRecordReader {

    private static Logger log = LoggerFactory.getLogger(TestMNISTRecordReader.class);


    @Test
    public void testMNISTRecordReadeR_InputStream() throws Exception {

        //private transient MnistManager man;
        final int NUM_EXAMPLES = 60000;
        String TEMP_ROOT = System.getProperty("user.home");
        String MNIST_ROOT = TEMP_ROOT + File.separator + "MNIST" + File.separator;

        String MNIST_Filename = MNIST_ROOT + MNISTRecordReader.TRAINING_FILES_FILENAME_UNZIPPED;

        // 1. check for the MNIST data first!

        // does it exist?

        // if not, then let's download it

        log.info("Checking to see if MNIST exists locally: " + MNIST_ROOT);

        if(!new File(MNIST_ROOT).exists()) {
            log.info("Downloading and unzipping the MNIST dataset locally to: " + MNIST_ROOT);
            new MnistFetcher().downloadAndUntar();
        } else {

            log.info("MNIST already exists locally...");

        }

        if ( new File(MNIST_Filename).exists() ) {
            log.info("The images file exists locally unzipped!");
        } else {
            log.info("The images file DOES NOT exist locally unzipped!");
        }


        // next, lets fire up the record reader and give it a whirl...

        RecordReader reader = new MNISTRecordReader();

        //ClassPathResource res = new ClassPathResource( MNIST_Filename );
        File resMNIST = new File( MNIST_Filename );
        InputStream targetStream = new FileInputStream( resMNIST );
        // resMNIST.

        reader.initialize(new InputStreamInputSplit(targetStream, resMNIST.toURI()));

        assertTrue(reader.hasNext());

        SVMLightRecordWriter writer = new SVMLightRecordWriter(new File("mnist_svmlight.txt"));

        while ( reader.hasNext() ) {
            Collection<Writable> record = reader.next();
            writer.write(record);
            // 784 pixels converted to one array entry + a class label entry == 2
            assertEquals( 2, record.size() );
            assertEquals( 784, ((ArrayWritable)(record.iterator().next())).length() );
        }

        writer.close();



        log.info("looped through 60k examples");

    }


}
