package org.deeplearning4j.datasets.fetchers;

import org.apache.commons.io.FileUtils;

import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.util.SerializationUtils;

import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * Curves data fetcher
 *
 * @author Adam Gibson
 */
public class CurvesDataFetcher extends BaseDataFetcher {

    public final static String CURVES_URL = "https://s3.amazonaws.com/dl4j-distribution/curves.ser";
    public final static String LOCAL_DIR_NAME =  "curves";
    public final static String CURVES_FILE_NAME = "curves.ser";
    private DataSet data;



    public CurvesDataFetcher() throws IOException {
        download();
        totalExamples= data.numExamples();


    }

    private void download() throws IOException {
        // mac gives unique tmp each run and we want to store this persist
        // this data across restarts
        File tmpDir = new File(System.getProperty("user.home"));

        File baseDir = new File(tmpDir, LOCAL_DIR_NAME);
        if(!(baseDir.isDirectory() || baseDir.mkdir())) {
            throw new IOException("Could not mkdir " + baseDir);
        }

        File dataFile = new File(baseDir,CURVES_FILE_NAME);

        if(!dataFile.exists() || !dataFile.isFile()) {
            log.info("Downloading curves dataset...");
            FileUtils.copyURLToFile(new URL(CURVES_URL), dataFile);
        }


        data = SerializationUtils.readObject(dataFile);




    }


    @Override
    public boolean hasMore() {
        return super.hasMore();
    }

    /**
     * Fetches the next dataset. You need to call this
     * to getFromOrigin a new dataset, otherwise {@link #next()}
     * just returns the last data applyTransformToDestination fetch
     *
     * @param numExamples the number of examples to fetch
     */
    @Override
    public void fetch(int numExamples) {
        if(cursor >= data.numExamples()) {
            cursor = data.numExamples();
        }

        curr = data.get(NDArrayIndex.interval(cursor, cursor + numExamples).indices());
        log.info("Fetched " + curr.numExamples());
        if(cursor + numExamples < data.numExamples())
            cursor += numExamples;
        //always stay at the end
        else if(cursor + numExamples > data.numExamples())
            cursor = data.numExamples()  -1;

    }
}
