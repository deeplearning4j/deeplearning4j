package org.deeplearning4j.datasets.rearrange;

import com.google.common.io.Files;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Rearrange an unstructured dataset
 * in to split test/train
 * on the file system
 *
 * @author Adam Gibson
 */
public class LocalUnstructuredDataFormatter {
    private File splitRootDir, rootDir;
    private File train, test;
    private LabelingType labelingType;
    private int numExamplesTotal = -1;
    private int numTestExamples = -1;
    private double percentTrain = 0.0;
    private int numExamplesToTrainOn = -1;
    private List<String> allFiles = new ArrayList<>();


    public enum LabelingType {
        NAME, DIRECTORY
    }


    /**
     *  @param destinationRootDir the destination root directory
     * @param rootDir the root directory of the original data
     * @param labelingType the labeling type to use (NAME/Label)
     * @param percentTrain the percent train to hold out
     */
    public LocalUnstructuredDataFormatter(File destinationRootDir, File rootDir, LabelingType labelingType,
                    double percentTrain) {
        this.percentTrain = percentTrain;
        this.rootDir = rootDir;
        splitRootDir = new File(destinationRootDir, "split");
        if (splitRootDir.exists())
            throw new IllegalStateException("Train/test split already exists");
        train = new File(splitRootDir, "train");
        test = new File(splitRootDir, "test");
        train.mkdirs();
        test.mkdirs();
        this.labelingType = labelingType;
    }

    public void rearrange() {
        //accumulate all files/data
        Iterator<File> files = FileUtils.iterateFiles(rootDir, null, true);
        while (files.hasNext()) {
            allFiles.add(files.next().getAbsolutePath());
        }

        numExamplesTotal = allFiles.size();
        //randomly partition the data; afterwards split in to test train
        //based on the current position in the files
        int numExampleForTrain = (int) (percentTrain * (double) numExamplesTotal);
        this.numExamplesToTrainOn = numExampleForTrain;
        this.numTestExamples = numExamplesTotal - numExampleForTrain;
        Collections.shuffle(allFiles);
        for (int i = 0; i < numExamplesTotal; i++) {
            String dir = getNewDestination(allFiles.get(i), i < numExampleForTrain);
            File origin = new File(allFiles.get(i));
            File newDir = new File(dir);
            newDir.getParentFile().mkdirs();
            try {
                Files.copy(origin, newDir);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

    }


    public String getNewDestination(String path, boolean train) {
        File file = new File(path);
        switch (labelingType) {
            case NAME:
                if (train) {
                    File dir = new File(this.train, getNameLabel(path));
                    File name = new File(dir, file.getName());
                    return name.getAbsolutePath();

                } else {
                    File dir = new File(this.test, getNameLabel(path));
                    File name = new File(dir, file.getName());
                    return name.getAbsolutePath();


                }
            case DIRECTORY:
                if (train) {
                    File dir = new File(this.train, getPathLabel(path));
                    File name = new File(dir, file.getName());
                    return name.getAbsolutePath();

                } else {
                    File dir = new File(this.test, getPathLabel(path));
                    File name = new File(dir, file.getName());
                    return name.getAbsolutePath();

                }
        }

        throw new IllegalStateException("Illegal labeling type ");
    }


    public String getPathLabel(String path) {
        return new File(path).getParentFile().getName();
    }


    public String getNameLabel(String path) {
        int startOfFormat = path.lastIndexOf('.');
        if (startOfFormat < 0)
            throw new IllegalStateException("Illegal path; no format found");
        StringBuilder label = new StringBuilder();
        while (path.charAt(startOfFormat) != '-') {
            label.append(path.charAt(startOfFormat));
            startOfFormat--;
        }

        if (startOfFormat < 0)
            throw new IllegalStateException("Illegal path; no - found. A dash is used to inidicate a lbale.");
        return label.reverse().toString();
    }

    public int getNumExamplesTotal() {
        return numExamplesTotal;
    }

    public int getNumExamplesToTrainOn() {
        return numExamplesToTrainOn;
    }

    public int getNumTestExamples() {
        return numTestExamples;
    }

    public File getTest() {
        return test;
    }

    public File getTrain() {
        return train;
    }



}
