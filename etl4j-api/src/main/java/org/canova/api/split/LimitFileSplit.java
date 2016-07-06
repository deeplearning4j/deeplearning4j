package org.canova.api.split;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URI;
import java.util.*;

/**
 * File input split that allows limits on number of files loaded.
 * Define total number of examples to pull from the dataset.
 * Use numCategories to split the numExamples and pull equal number of examples per category.
 * Use pattern and patternPosition to segment the filename that represents the category.
 *
 * Created by nyghtowl on 11/9/15.
 */
@Deprecated
public class LimitFileSplit extends FileSplit {

    protected static Logger log = LoggerFactory.getLogger(LimitFileSplit.class);
    protected int totalNumExamples;
    protected int numCategories;
    protected String pattern; // Pattern to split and segment file name, pass in regex
    protected int patternPosition = 0;

    public LimitFileSplit(File rootDir, String[] allowFormat, boolean recursive, int numExamples, int numCategories, String pattern, int patternPosition, Random random) {
        super(rootDir, allowFormat, recursive, random, false);
        this.totalNumExamples = numExamples;
        this.numCategories = numCategories;
        this.pattern = pattern;
        this.patternPosition = patternPosition;
        this.initialize();
    }

    public LimitFileSplit(File rootDir, int numExamples) {
        this(rootDir, null, true, numExamples, 1, null, 0, null);
    }

    public LimitFileSplit(File rootDir, String[] allowFormat, boolean recursive, int numExamples) {
        this(rootDir, allowFormat, recursive, numExamples, 1, null, 0, null);
    }

    public LimitFileSplit(File rootDir, int numExamples, String pattern) {
        this(rootDir, null, true, numExamples, 1, pattern, 0, null);
    }

    public LimitFileSplit(File rootDir, int numExamples, int numCategories, String pattern) {
        this(rootDir, null, true, numExamples, numCategories, pattern, 0, null);
    }

    public LimitFileSplit(File rootDir, String[] allowFormat, int numExamples, int numCategories, String pattern) {
        this(rootDir, allowFormat, true, numExamples, numCategories, pattern, 0, null);
    }

    public LimitFileSplit(File rootDir, String[] allowFormat, int numExamples, int numCategories, String pattern, int patternPosition, Random random) {
        this(rootDir, allowFormat, true, numExamples, numCategories, pattern, patternPosition, random);
    }

    public LimitFileSplit(File rootDir, int numExamples, Random random) {
        this(rootDir, null, true, numExamples, 1, null, 0, random);
    }

    public LimitFileSplit(File rootDir, int numExamples, String pattern, Random random) {
        this(rootDir, null, true, numExamples, 1, pattern, 0, random);
    }

    public LimitFileSplit(File rootDir, String[] allowFormat, int numExamples, int numCategories, String pattern, Random random) {
        this(rootDir, allowFormat, true, numExamples, numCategories, pattern, 0, random);
    }

    public LimitFileSplit(File rootDir, String[] allowFormat, int numExamples, String pattern, Random random) {
        this(rootDir, allowFormat, true, numExamples, 1, pattern, 0, random);
    }

    public LimitFileSplit(File rootDir, String[] allowFormat, int numExamples, String pattern, int patternPosition, Random random) {
        this(rootDir, allowFormat, true, numExamples, 1, pattern, patternPosition, random);
    }

    @Override
    protected void initialize() {
        Collection<File> subFiles;

        // Limits number files listed will pull set number from each directory
        Iterator iter = FileUtils.iterateFiles(rootDir, allowFormat, recursive);
        subFiles = new ArrayList<>();

        int numExamplesPerCategory = (totalNumExamples >= numCategories) ? (totalNumExamples / numCategories) + (totalNumExamples % numCategories): 1;

        File file;
        Map<String, Integer> fileCount = new HashMap<>();
        String currName = "";
        int totalCount = 0;
        int numCategoryCount = 0;

        while (iter.hasNext() && numCategoryCount <= numCategories) {
            if(totalCount >= totalNumExamples) break;
            file = (File) iter.next();
            if (pattern != null) {
                // Label is in the filename
                currName = FilenameUtils.getBaseName(file.getName()).split(pattern)[patternPosition];
            } else {
                // Label is in the directory
                currName = FilenameUtils.getBaseName(file.getParent());
            }

            if (file.isFile()){

                if (!fileCount.containsKey(currName)) {
                    fileCount.put(currName, 0);
                    numCategoryCount++;
                }
                int categoryCount = fileCount.get(currName);

                if (categoryCount < numExamplesPerCategory) {
                    subFiles.add(file);
                    fileCount.put(currName, categoryCount + 1);
                    totalCount++;

                }
            }
            if (numExamplesPerCategory == 0) log.info("{} number of categories were loaded", fileCount.keySet().size() );
        }

        locations = new URI[subFiles.size()];

        if (randomize) Collections.shuffle((List<File>) subFiles, random);
        int count = 0;

        for (File f : subFiles) {
            if (f.getPath().startsWith("file:"))
                locations[count++] = URI.create(f.getPath());
            else
                locations[count++] = f.toURI();
            length += f.length();
        }
    }

}
