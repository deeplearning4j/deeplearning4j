package org.deeplearning4j.keras;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.hdf5;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.nio.file.Path;

import static org.bytedeco.javacpp.hdf5.H5F_ACC_RDONLY;

/**
 * Reads and INDArray from a HDF5 dataset. The array is expected to be included as the "data" dataset inside the file.
 * The shape of the output array is the same as the one stored in the HDF5 file.
 *
 * @author pkoperek@gmail.com
 */
public class NDArrayHDF5Reader {

    /**
     * Reads an HDF5 file into an NDArray.
     *
     * @param inputFilePath Path of the HDF5 file
     * @return NDArray with data and a correct shape
     */
    public INDArray readFromPath(Path inputFilePath) {
        try (hdf5.H5File h5File = new hdf5.H5File()) {
            h5File.openFile(inputFilePath.toString(), H5F_ACC_RDONLY);
            hdf5.DataSet dataSet = h5File.asCommonFG().openDataSet("data");
            int[] shape = extractShape(dataSet);
            long totalSize = ArrayUtil.prodLong(shape);
            DataBuffer dataBuffer = readFromDataSet(dataSet, (int) totalSize);

            INDArray input = Nd4j.create(shape);
            input.setData(dataBuffer);

            return input;
        }
    }

    private DataBuffer readFromDataSet(hdf5.DataSet dataSet, int total) {
        float[] dataBuffer = new float[total];
        FloatPointer fp = new FloatPointer(dataBuffer);
        dataSet.read(fp, new hdf5.DataType(hdf5.PredType.NATIVE_FLOAT()));
        fp.get(dataBuffer);
        return Nd4j.createBuffer(dataBuffer);
    }

    private int[] extractShape(hdf5.DataSet dataSet) {
        hdf5.DataSpace space = dataSet.getSpace();
        int nbDims = space.getSimpleExtentNdims();
        long[] shape = new long[nbDims];
        space.getSimpleExtentDims(shape);
        return ArrayUtil.toInts(shape);
    }
}
