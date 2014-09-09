package org.deeplearning4j.models.word2vec.datasets.loader;

import static org.junit.Assert.*;

import org.deeplearning4j.datasets.loader.ReutersNewsGroupsLoader;
import org.junit.Test;

import java.io.File;

/**
 * Adam Gibson
 */
public class ReutersLoaderTest {
    @Test
    public void testReutersLoad() throws Exception {
        ReutersNewsGroupsLoader loader = new ReutersNewsGroupsLoader(true);
        String homeDir = System.getProperty("user.home");
        File rootDir = new File(homeDir,"reuters");
        assertEquals(true,rootDir.exists());
        assertEquals(20, rootDir.listFiles().length);
    }

}
