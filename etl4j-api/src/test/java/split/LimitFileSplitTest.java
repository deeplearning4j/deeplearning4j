package split;

import org.canova.api.split.InputSplit;
import org.canova.api.split.LimitFileSplit;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.regex.Pattern;

import static org.junit.Assert.assertEquals;

/**
 * Created by nyghtowl on 11/9/15.
 */
public class LimitFileSplitTest {

    protected File file1, file2, file3, file4, file5, file6, newPath;
    protected String[] allForms = {"jpg", "jpeg", "JPG", "JPEG"};
    protected static String localPath = System.getProperty("java.io.tmpdir") + File.separator;
    protected static String testPath = localPath + "test" + File.separator;

    // These cannot run on TravisCI - uncomment when checking locally

//    @Before
//    public void doBefore() throws IOException {
//        newPath = new File(testPath);
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
//    public void testInitializeLimitedFiles() throws IOException{
//        InputSplit split = new LimitFileSplit(newPath, 4);
//        assertEquals(4, split.locations().length);
//        assertEquals(file1.toURI(), split.locations()[0]);
//        assertEquals(file3.toURI(), split.locations()[2]);
//
//    }
//
//    @Test
//    public void testInitializeLimitedNoCategoryFilesPattern() throws IOException {
//        InputSplit split = new LimitFileSplit(newPath, 4, Pattern.quote("_"));
//        assertEquals(4, split.locations().length, 0);
//        assertEquals(file1.toURI(), split.locations()[0]);
//        assertEquals(file3.toURI(), split.locations()[2]);
//    }
//
//    @Test
//    public void testInitializeLimitedCategoryFilesPattern() throws IOException {
//        InputSplit split = new LimitFileSplit(newPath, 2, 2, Pattern.quote("_"));
//        assertEquals(2, split.locations().length);
//        assertEquals(file1.toURI(), split.locations()[0]);
//        assertEquals(file4.toURI(), split.locations()[1]);
//    }
//
//    @Test
//    public void testInitializeLimitedAllowedCategoryFilesPattern() throws IOException {
//        InputSplit split = new LimitFileSplit(newPath, allForms, 2, 2, Pattern.quote("_"));
//        assertEquals(2, split.locations().length);
//        assertEquals(file1.toURI(), split.locations()[0]);
//        assertEquals(file6.toURI(), split.locations()[1]);
//    }
//
//    @Test
//    public void testInitializeLimitedAllowedCategoryFilesPatternPosition() throws IOException {
//
//        File newPath2 = new File(localPath + "test2" + File.separator);
//        newPath2.mkdir();
//
//        File filea = File.createTempFile("4_myfile_1", ".jpg", newPath2);
//        File fileb = File.createTempFile("5_myfile_2", ".txt", newPath2);
//        File filec = File.createTempFile("6_myfile_3", ".jpg", newPath2);
//        File filed = File.createTempFile("7_treehouse_4", ".csv", newPath2);
//        File filee = File.createTempFile("8_treehouse_5", ".jpg", newPath2);
//        File filef = File.createTempFile("9_treehouse_6", ".jpg", newPath2);
//
//        InputSplit split = new LimitFileSplit(newPath2, allForms, 4, 2, Pattern.quote("_"), 1, null);
//        assertEquals(4, split.locations().length);
//        assertEquals(filea.toURI(), split.locations()[0]);
//        assertEquals(filee.toURI(), split.locations()[2]);
//
//        filea.deleteOnExit();
//        fileb.deleteOnExit();
//        filec.deleteOnExit();
//        filed.deleteOnExit();
//        filee.deleteOnExit();
//        filef.deleteOnExit();
//        newPath2.deleteOnExit();
//
//    }

//    @After
//    public void doAfter(){
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
