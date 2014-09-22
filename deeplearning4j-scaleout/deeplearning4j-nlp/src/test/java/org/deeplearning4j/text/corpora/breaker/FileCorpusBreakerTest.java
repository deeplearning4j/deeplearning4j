package org.deeplearning4j.text.corpora.breaker;

import org.junit.Test;

import java.io.File;
import java.io.FileOutputStream;
import java.net.URI;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 9/21/14.
 */
public class FileCorpusBreakerTest {



    @Test
    public void testFileBreaker() throws Exception {
        File f = new File("a.bin");
        f.createNewFile();
        FileOutputStream s = new FileOutputStream("a.bin");
        byte[] buf = new byte[1024];
        s.write(buf);
        s.flush();
        s.close();

        CorpusBreaker fileBreaker = new FileCorpusBreaker(new File("a.bin"),1,1024);
        URI[] locations = fileBreaker.corporaLocations();
        int numFiles = 1024;
        assertEquals(numFiles,locations.length);
        new File("a.bin").delete();
        fileBreaker.cleanup();



    }
}
