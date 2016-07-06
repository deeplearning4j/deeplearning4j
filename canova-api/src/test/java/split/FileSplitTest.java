package split;

import org.canova.api.split.InputSplit;
import org.junit.After;
import org.junit.rules.TemporaryFolder;
import org.canova.api.split.FileSplit;

import java.io.File;
import java.io.IOException;
import java.util.Random;


import static org.junit.Assert.*;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;


/**
 * Created by nyghtowl on 11/8/15.
 */
public class FileSplitTest {

    protected File file, file1, file2, file3, file4, file5, file6, newPath;
    protected String[] allForms = {"jpg", "jpeg", "JPG", "JPEG"};
    private static String localPath = System.getProperty("java.io.tmpdir") + File.separator;
    private static String testPath = localPath + "test" + File.separator;

    // These cannot run on TravisCI - uncomment when checking locally

    @Rule
    public TemporaryFolder mainFolder = new TemporaryFolder();

//
//    @Before
//    public void doBefore() throws IOException {
//        file = mainFolder.newFile("myfile.txt");
//
//        newPath = new File(testPath);
//
//        newPath.mkdir();
//
//        file1 = File.createTempFile("myfile_1", ".jpg", newPath);
//        file2 = File.createTempFile("myfile_2", ".txt", newPath);
//        file3 = File.createTempFile("myfile_3", ".jpg", newPath);
//        file4 = File.createTempFile("treehouse_4", ".csv", newPath);
//        file5 = File.createTempFile("treehouse_5", ".csv", newPath);
//        file6 = File.createTempFile("treehouse_6", ".jpg", newPath);
//
//    }
//
//    @Test
//    public void testInitializeLoadSingleFile(){
//        InputSplit split = new FileSplit(file, allForms);
//        assertEquals(split.locations()[0], file.toURI());
//
//    }
//
//    @Test
//    public void testInitializeLoadMulFiles() throws IOException{
//        InputSplit split = new FileSplit(newPath, allForms, true);
//        assertEquals(3, split.locations().length);
//        assertEquals(file1.toURI(), split.locations()[0]);
//        assertEquals(file3.toURI(), split.locations()[1]);
//    }
//
//    @Test
//    public void testInitializeMulFilesShuffle() throws IOException{
//        InputSplit split = new FileSplit(newPath, new Random(123));
//        InputSplit split2 = new FileSplit(newPath, new Random(123));
//        assertEquals(6, split.locations().length);
//        assertEquals(6, split2.locations().length);
//        assertEquals(split.locations()[3], split2.locations()[3]);
//    }
//
//    @After
//    public void doAfter(){
//        mainFolder.delete();
//        file.delete();
//        file1.delete();
//        file2.delete();
//        file3.delete();
//        file4.delete();
//        file5.delete();
//        file6.delete();
//        newPath.delete();
//
//    }

}
