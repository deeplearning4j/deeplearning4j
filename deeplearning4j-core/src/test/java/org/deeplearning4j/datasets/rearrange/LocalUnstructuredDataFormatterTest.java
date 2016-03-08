package org.deeplearning4j.datasets.rearrange;

import org.apache.commons.io.FileUtils;
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.util.Iterator;

import static org.junit.Assert.*;


/**
 * Created by agibsonccc on 9/11/15.
 */
public class LocalUnstructuredDataFormatterTest {

    @Test
    @Ignore
    public void testRearrange() throws Exception {
        //ensure exists
        File destinationDir = new File(System.getProperty("user.home"),"rearrangedlfw");
        if(destinationDir.exists())
            FileUtils.deleteDirectory(destinationDir);
        LocalUnstructuredDataFormatter formatter = new LocalUnstructuredDataFormatter(destinationDir,new File(System.getProperty("user.home"),"lfw"), LocalUnstructuredDataFormatter.LabelingType.DIRECTORY,0.8);
        formatter.rearrange();
        //train and test in the split directory
        File splitRoot = new File(destinationDir,"split");
        assertEquals(2,splitRoot.listFiles().length);
        Iterator<File> files = FileUtils.iterateFiles(splitRoot,null,true);
        int count = 0;
        while(files.hasNext()) {
            files.next();
            count++;
        }

        Iterator<File> trainFiles = FileUtils.iterateFiles(new File(splitRoot,"train"),null,true);
        int trainFilesCount = 0;
        while(trainFiles.hasNext()) {
            trainFiles.next();
            trainFilesCount++;
        }

        //assert the number of files is the same in the split test train as the original dataset
        assertEquals(formatter.getNumExamplesTotal(),count);
        assertEquals(formatter.getNumExamplesToTrainOn(),trainFilesCount);

        Iterator<File> testFiles = FileUtils.iterateFiles(new File(splitRoot,"test"),null,true);
        int testFilesCount = 0;
        while(testFiles.hasNext()) {
            testFiles.next();
            testFilesCount++;
        }
        assertEquals(formatter.getNumTestExamples(),testFilesCount);

    }

}
