package org.deeplearning4j.datasets.savers;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FilenameUtils;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import java.io.File;

/**
 * Created by kepricon on 17. 4. 19.
 */
@Slf4j
public class DataSetSaver {
    private static final int NUM_CORES = Runtime.getRuntime().availableProcessors();
    private static final int numThreads = NUM_CORES * 2;
    private static final ExecutorService taskExecutor = Executors.newFixedThreadPool(numThreads);

    public static void saveDataSets(DataSetIterator iter, String savePath) {

        File dest = new File(savePath);
        if (dest.exists()) {
            log.info(savePath + " exist, delete it and retry");
        } else {
            dest.mkdirs();
            log.info("datasets will be saved at " + savePath);

            log.info("Loading data...");

            AtomicInteger counter = new AtomicInteger(0);

            while (iter.hasNext()) {
                String filePath = FilenameUtils.concat(savePath, "dataset-" + (counter.getAndIncrement()) + ".bin");
                taskExecutor.execute(new Worker(counter.get() % numThreads, iter.next(), new File(filePath)));

                if (counter.get() % 100 == 0) {
                    log.info("{} datasets queued so far...", counter.get());
                }
            }

            taskExecutor.shutdown();

            try {
                taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                log.warn("Unable to terminate executors", e);
            }
        }
    }

    private static class Worker extends Thread {
        int id;
        DataSet ds;
        File path;

        public Worker(int id, DataSet ds, File path) {
            this.id = id;
            this.ds = ds;
            this.path = path;
        }

        public void run() {
            ds.save(path);
            log.debug("[" + id + "] thread saved : " + path.getAbsolutePath());
        }
    }
}
