package org.deeplearning4j;

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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

import static org.bytedeco.javacpp.hdf5.H5F_ACC_RDONLY;

public class DeepLearning4jEntryPoint {

    private static final Logger logger = LoggerFactory.getLogger(DeepLearning4jEntryPoint.class);

    public DeepLearning4jEntryPoint() {

    }

    public void fit(
            String modelFilePath,
            String type,
            String trainFeaturesFile,
            String trainLabelsFile,
            int batchSize,
            long nbEpoch,
            String validationXFilePath,
            String validationYFilePath,
            String dimOrdering

    ) throws IOException, InterruptedException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
        try {
            MultiLayerNetwork multiLayerNetwork;
            if ("sequential".equals(type)) {
                multiLayerNetwork = KerasModelImport.importKerasSequentialModelAndWeights(modelFilePath);
                multiLayerNetwork.init();
            } else {
                throw new RuntimeException("Model type unsupported! (" + type + ")");
            }

            INDArray features = h5FileToNDArray(trainFeaturesFile);
            INDArray labels = h5FileToNDArray(trainLabelsFile);

            INDArrayIndex[] ndIndexes = new INDArrayIndex[features.shape().length];
            for (int i = 0; i < ndIndexes.length; i++) {
                ndIndexes[i] = NDArrayIndex.all();
            }

            for (int i = 0; i < nbEpoch; i++) {
                logger.info("Fitting: " + i);

                int begin = 0;
                while (begin < features.size(0)) {
                    int end = begin + batchSize;

                    if(logger.isTraceEnabled()) {
                        logger.trace("Processing batch: " + begin + " " + end);
                    }

                    ndIndexes[0] = NDArrayIndex.interval(begin, end);
                    INDArray featuresBatch = features.get(ndIndexes);
                    INDArray labelsBatch = labels.get(NDArrayIndex.interval(begin, end));
                    multiLayerNetwork.fit(featuresBatch, labelsBatch);

                    begin += batchSize;
                }
            }

            logger.info("Learning model finished");
        } catch (Throwable e) {
            logger.error("Error while handling request!", e);
            throw e;
        }
    }

    private INDArray h5FileToNDArray(String inputFilePath) {
        hdf5.H5File h5File = new hdf5.H5File();
        h5File.openFile(inputFilePath, H5F_ACC_RDONLY);
        hdf5.DataSet dataSet = h5File.asCommonFG().openDataSet("data");
        long[] shape = extractShape(dataSet);
        long totalSize = mulArray(shape);
        float[] dataBuffer = readFromDataSet(dataSet, (int) totalSize);

        INDArray input = Nd4j.create(toInt(shape));
        new RecursiveCopier(input, dataBuffer, shape).copy();

        h5File.close();

        return input;
    }

    private float[] readFromDataSet(hdf5.DataSet dataSet, int total) {
        float[] dataBuffer = new float[total];
        FloatPointer fp = new FloatPointer(dataBuffer);
        dataSet.read(fp, new hdf5.DataType(hdf5.PredType.NATIVE_FLOAT()));
        fp.get(dataBuffer);
        return dataBuffer;
    }

    private long mulArray(long[] shape) {
        long retVal = 1L;
        for (int i = 0; i < shape.length; i++) {
            retVal *= shape[i];
        }
        return retVal;
    }

    private long[] extractShape(hdf5.DataSet dataSet) {
        hdf5.DataSpace space = dataSet.getSpace();
        int nbDims = space.getSimpleExtentNdims();
        long[] shape = new long[nbDims];
        space.getSimpleExtentDims(shape);
        return shape;
    }

    private int[] toInt(long[] array) {
        int[] retVal = new int[array.length];

        for (int i = 0; i < array.length; i++) {
            retVal[i] = (int) array[i];
        }

        return retVal;
    }

}
