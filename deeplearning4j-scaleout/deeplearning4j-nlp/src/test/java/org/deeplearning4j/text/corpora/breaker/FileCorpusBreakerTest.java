package org.deeplearning4j.text.corpora.breaker;

import org.junit.Test;

import java.io.File;
import java.io.RandomAccessFile;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 9/21/14.
 */
public class FileCorpusBreakerTest {



    @Test
    public void testFileBreaker() throws Exception {
        RandomAccessFile f = new RandomAccessFile("t", "rw");
        f.setLength(1024 * 1024 * 1024);

        CorpusBreaker fileBreaker = new FileCorpusBreaker(new File("t"),1000,1000000);
        


    }
}
