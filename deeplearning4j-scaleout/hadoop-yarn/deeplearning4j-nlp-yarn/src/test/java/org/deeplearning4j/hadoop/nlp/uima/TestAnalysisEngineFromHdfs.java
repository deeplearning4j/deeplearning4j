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
