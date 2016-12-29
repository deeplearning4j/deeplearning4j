package org.deeplearning4j.keras;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.hdf5;
import org.deeplearning4j.nn.modelimport.keras.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.IOException;

import static org.bytedeco.javacpp.hdf5.H5F_ACC_RDONLY;

/**
 * API exposed to the Python side. This class contains methods which are used by the python wrapper.
 * It is instantiated directly in the server code.
 */
@Slf4j
public class DeepLearning4jEntryPoint {

    /**
     * Performs fitting of the model which is referenced in the parameters according to learning parameters specified.
     *
     * @param entryPointFitParameters
     * @throws IOException
     * @throws InterruptedException
     * @throws InvalidKerasConfigurationException
     * @throws UnsupportedKerasConfigurationException
     */
    public void fit(EntryPointFitParameters entryPointFitParameters)
            throws IOException, InterruptedException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {

        try {
            MultiLayerNetwork multiLayerNetwork;
            if (KerasModelType.SEQUENTIAL.equals(entryPointFitParameters.getType())) {
                multiLayerNetwork = KerasModelImport.importKerasSequentialModelAndWeights(entryPointFitParameters.getModelFilePath());
                multiLayerNetwork.init();
            } else {
                throw new RuntimeException("Model type unsupported! (" + entryPointFitParameters.getType() + ")");
            }

            INDArray features = h5FileToNDArray(entryPointFitParameters.getTrainFeaturesFile());
            INDArray labels = h5FileToNDArray(entryPointFitParameters.getTrainLabelsFile());
            int batchSize = entryPointFitParameters.getBatchSize();

            for (int i = 0; i < entryPointFitParameters.getNbEpoch(); i++) {
                log.info("Fitting: " + i);

                fitInBatches(multiLayerNetwork, features, labels, batchSize);
            }

            log.info("Learning model finished");
        } catch (Throwable e) {
            log.error("Error while handling request!", e);
            throw e;
        }
    }

    void fitInBatches(MultiLayerNetwork multiLayerNetwork, INDArray features, INDArray labels, int batchSize) {
        final INDArrayIndex[] ndIndexes = createSlicingIndexes(features.shape().length);

        int begin = 0;

        while (begin < features.size(0)) {
            int end = begin + batchSize;

            if (log.isTraceEnabled()) {
                log.trace("Processing batch: " + begin + " " + end);
            }

            ndIndexes[0] = NDArrayIndex.interval(begin, end);
            INDArray featuresBatch = features.get(ndIndexes);
            INDArray labelsBatch = labels.get(NDArrayIndex.interval(begin, end));
            multiLayerNetwork.fit(featuresBatch, labelsBatch);

            begin += batchSize;
        }
    }

    private INDArrayIndex[] createSlicingIndexes(int length) {
        INDArrayIndex[] ndIndexes = new INDArrayIndex[length];
        for (int i = 0; i < ndIndexes.length; i++) {
            ndIndexes[i] = NDArrayIndex.all();
        }
        return ndIndexes;
    }

    private INDArray h5FileToNDArray(String inputFilePath) {
        try (hdf5.H5File h5File = new hdf5.H5File()) {
            h5File.openFile(inputFilePath, H5F_ACC_RDONLY);
            hdf5.DataSet dataSet = h5File.asCommonFG().openDataSet("data");
            int[] shape = extractShape(dataSet);
            long totalSize = ArrayUtil.prodLong(shape);
            float[] dataBuffer = readFromDataSet(dataSet, (int) totalSize);

            INDArray input = Nd4j.create(shape);
            new RecursiveCopier(input, dataBuffer, shape).copy();
            return input;
        }
    }

    private float[] readFromDataSet(hdf5.DataSet dataSet, int total) {
        float[] dataBuffer = new float[total];
        FloatPointer fp = new FloatPointer(dataBuffer);
        dataSet.read(fp, new hdf5.DataType(hdf5.PredType.NATIVE_FLOAT()));
        fp.get(dataBuffer);
        return dataBuffer;
    }

    private int[] extractShape(hdf5.DataSet dataSet) {
        hdf5.DataSpace space = dataSet.getSpace();
        int nbDims = space.getSimpleExtentNdims();
        long[] shape = new long[nbDims];
        space.getSimpleExtentDims(shape);
        return ArrayUtil.toInts(shape);
    }

}
