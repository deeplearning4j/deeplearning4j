package org.nd4j.linalg.dataset;

import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Mini batch file datasetiterator
 * auto partitions a dataset in to mini batches
 */
public class MiniBatchFileDataSetIterator implements DataSetIterator {
    private int batchSize;
    private List<String[]> paths;
    private int currIdx;
    private File rootDir;
    private int totalExamples;
    private int totalLabels;
    private int totalBatches = -1;
    private DataSetPreProcessor dataSetPreProcessor;



    /**
     *
     * @param baseData the base dataset
     * @param batchSize the batch size to split by
     * @throws IOException
     */
    public MiniBatchFileDataSetIterator(DataSet baseData,int batchSize) throws IOException {
        this(baseData,batchSize,true);

    }
    /**
     *
     * @param baseData the base dataset
     * @param batchSize the batch size to split by
     * @throws IOException
     */
    public MiniBatchFileDataSetIterator(DataSet baseData,int batchSize,boolean delete,File rootDir) throws IOException {
        this.batchSize = batchSize;
        this.rootDir = new File(rootDir,UUID.randomUUID().toString());
        this.rootDir.mkdirs();
        if(delete)
            Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        FileUtils.deleteDirectory(MiniBatchFileDataSetIterator.this.rootDir);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }));
        currIdx = 0;
        paths = new ArrayList<>();
        totalExamples = baseData.numExamples();
        totalLabels = baseData.numOutcomes();
        int offset = 0;
        totalBatches = baseData.numExamples() / batchSize;
        for(int i = 0; i < baseData.numExamples() / batchSize; i++) {
            paths.add(writeData(new DataSet(baseData.getFeatureMatrix().get(NDArrayIndex.interval(offset, offset + batchSize))
                    , baseData.getLabels().get(NDArrayIndex.interval(offset, offset + batchSize)))));
            offset += batchSize;
            if(offset >= totalExamples)
                break;
        }
    }

    /**
     *
     * @param baseData the base dataset
     * @param batchSize the batch size to split by
     * @throws IOException
     */
    public MiniBatchFileDataSetIterator(DataSet baseData,int batchSize,boolean delete) throws IOException {
        this(baseData, batchSize, delete, new File(System.getProperty("java.io.tmpdir")));
    }

    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException("Unable to load custom number of examples");
    }

    @Override
    public int totalExamples() {
        return totalExamples;
    }

    @Override
    public int inputColumns() {
        return 0;
    }

    @Override
    public int totalOutcomes() {
        return totalLabels;
    }

    @Override
    public void reset() {
        currIdx = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return currIdx;
    }

    @Override
    public int numExamples() {
        return totalExamples;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        this.dataSetPreProcessor = preProcessor;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return currIdx < totalBatches;
    }

    @Override
    public void remove() {
        //no opt;
    }

    @Override
    public DataSet next() {
        try {
            DataSet ret =  read(currIdx);
            if(dataSetPreProcessor != null)
                dataSetPreProcessor.preProcess(ret);
            currIdx++;

            return ret;
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read dataset");
        }
    }

    private DataSet read(int idx) throws IOException {
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(paths.get(idx)[0]));
        DataInputStream dis = new DataInputStream(bis);
        BufferedInputStream labelInputStream = new BufferedInputStream(new FileInputStream(paths.get(idx)[1]));
        DataInputStream labelDis = new DataInputStream(labelInputStream);
        DataSet d = new DataSet(Nd4j.read(dis),Nd4j.read(labelDis));
        dis.close();
        labelDis.close();
        return d;
    }


    private String[] writeData(DataSet write) throws IOException {
        String[] ret = new String[2];
        String dataSetId = UUID.randomUUID().toString();
        BufferedOutputStream dataOut = new BufferedOutputStream(new FileOutputStream(new File(rootDir,dataSetId + ".bin")));
        DataOutputStream dos = new DataOutputStream(dataOut);
        Nd4j.write(write.getFeatureMatrix(),dos);
        dos.flush();
        dos.close();


        BufferedOutputStream dataOutLabels = new BufferedOutputStream(new FileOutputStream(new File(rootDir,dataSetId + ".labels.bin")));
        DataOutputStream dosLabels = new DataOutputStream(dataOutLabels);
        Nd4j.write(write.getLabels(), dosLabels);
        dosLabels.flush();
        dos.close();
        ret[0] = new File(rootDir,dataSetId + ".bin").getAbsolutePath();
        ret[1] = new File(rootDir,dataSetId + ".labels.bin").getAbsolutePath();
        return ret;

    }

    public File getRootDir() {
        return rootDir;
    }

    public void setRootDir(File rootDir) {
        this.rootDir = rootDir;
    }
}
