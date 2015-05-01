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

package org.deeplearning4j.hadoop.nlp.uima;

import static org.junit.Assert.*;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.deeplearning4j.hadoop.nlp.text.ConfigurableSentenceIterator;
import org.deeplearning4j.text.annotator.SentenceAnnotator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.io.IOException;

/**
 * Created by agibsonccc on 1/29/15.
 */
public class TestAnalysisEngineFromHdfs {

    private Configuration conf;
    private FileSystem fs;
    @Before
    public void before() throws IOException {
        conf = new Configuration();
        conf.set("fs.defaultFS", "file:///");
        File parentDir = new File("parent");
        parentDir.mkdir();
        FileUtils.writeStringToFile(new File(parentDir, "touch"), "hello");
        conf.set(ConfigurableSentenceIterator.ROOT_PATH,parentDir.toURI().toString());
        fs = FileSystem.get(conf);


    }

    @After
    public void after() throws IOException {
        FileUtils.deleteDirectory(new File("parent"));
        fs.close();
    }

    @Test
    public void testReadWrite() throws Exception {
        AnalysisEngine a = UimaSentenceIterator.segmenter();
        Path descriptor = new Path(new File("parent").toURI().toString(),"descriptor.xml");
        AnalysisEngineDescription desc = SentenceAnnotator.getDescription();
        AnalysisEngineHdfs.writeAnalysisEngineDescriptor(fs,descriptor,desc);
        AnalysisEngine a2 = AnalysisEngineHdfs.readConfFrom(fs,descriptor);


    }

}
