package org.deeplearning4j.datasets.fetchers;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.image.transform.ImageTransform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Fetcher for UCI synthetic control chart time series dataset.
 *
 * Details:     https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series
 * Data:        https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data
 * Image:       https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg
 *
 * @author Briton Park (bpark738)
 */
public class UciSequenceDataFetcher extends CacheableExtractableDataSetFetcher {

    private File fileDir;
    private static Logger log = LoggerFactory.getLogger(UciSequenceDataFetcher.class);

    public static int NUM_LABELS = 6;
    public static int NUM_EXAMPLES = NUM_LABELS * 100;

    @Override
    public String remoteDataUrl() {
        return "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";
    }

    @Override
    public String remoteDataUrl(DataSetType type) {
        return remoteDataUrl();
    }

    @Override
    public String localCacheName() {
        return "UCISequence_6";
    }

    @Override
    public long expectedChecksum() {
        return 104392751L;
    }

    @Override
    public long expectedChecksum(DataSetType type) {
        return expectedChecksum();
    }

    @Override
    public CSVSequenceRecordReader getRecordReader(long rngSeed, int[] shape, DataSetType set, ImageTransform transform) {
        return getRecordReader(rngSeed, set);
    }

    public CSVSequenceRecordReader getRecordReader(long rngSeed, DataSetType set) {

        // check empty cache
        if (LOCAL_CACHE.exists()) {
            if (LOCAL_CACHE.listFiles().length < 1) LOCAL_CACHE.delete();
        }

        try {
            if (!LOCAL_CACHE.exists()) downloadAndExtract();
        } catch (Exception e) {
            throw new RuntimeException("Could not download UCI Sequence data", e);
        }

        File dataPath;

        switch (set) {
            case TRAIN:
                dataPath = new File(LOCAL_CACHE, "/train");
                break;
            case TEST:
                dataPath = new File(LOCAL_CACHE, "/test");
                break;
            case VALIDATION:
                throw new IllegalArgumentException("You will need to manually iterate the directory, UCISequence data does not provide labels");

            default:
                dataPath = new File(LOCAL_CACHE, "/train");
        }

        try {
            downloadUCIData(dataPath);
            CSVSequenceRecordReader data;
            switch (set) {
                case TRAIN:
                    data = new CSVSequenceRecordReader(0, ", ");
                    data.initialize(new NumberedFileInputSplit(dataPath.getAbsolutePath() + "/%d.csv", 0, 449));
                    break;
                case TEST:
                    data = new CSVSequenceRecordReader(0, ", ");
                    data.initialize(new NumberedFileInputSplit(dataPath.getAbsolutePath() + "/%d.csv", 450, 599));
                    break;
                default:
                    data = new CSVSequenceRecordReader(0, ", ");
                    data.initialize(new NumberedFileInputSplit(dataPath.getAbsolutePath() + "/%d.csv", 0, 449));
            }

            return data;
        } catch (Exception e) {
            throw new RuntimeException("Could not process UCI data", e);
        }
    }

    private static void downloadUCIData(File dataPath) throws Exception {
        //if (dataPath.exists()) return;

        String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";
        String data = IOUtils.toString(new URL(url), Charset.defaultCharset());
        String[] lines = data.split("\n");

        int lineCount = 0;
        int index = 0;

        ArrayList<String> linesList = new ArrayList<>();

        for (String line : lines) {

            // label value
            Integer count = new Integer(lineCount++ / 100);

            // replace white space with commas and label value + new line
            line = line.replaceAll("\\s+", ", " + count.toString() + "\n");

            // add label to last number
            line = line + ", " + count.toString();
            linesList.add(line);
        }

        // randomly shuffle data
        Collections.shuffle(linesList, new Random(12345));

        for (String line : linesList) {
            File outPath = new File(dataPath, index + ".csv");
            FileUtils.writeStringToFile(outPath, line, Charset.defaultCharset());
            index++;
        }
    }
}
