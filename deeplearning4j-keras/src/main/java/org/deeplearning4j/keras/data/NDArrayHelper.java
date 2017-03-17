package org.deeplearning4j.keras.data;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Helper for converting INDArrays to byte[]. This is helpful for returning data to the
 * python environment in a py4j-friendly way. byte[] is converted to bytearray by
 * py4j and transmitted using a binary protocol.
 */
@Data
public class NDArrayHelper {

  public static double[][] toFlattened(INDArray array) {
    INDArray cArray = array.dup('c');
    int[] shape = cArray.shape();
    double[] data = cArray.data().asDouble();

    // convert shape to double so we can return it in a single
    // array for numpy
   double[] dShape = new double[shape.length];

    for(int i = 0; i < dShape.length; i++) {
      dShape[i] = (double) shape[i];
    }

    return new double[][]{data, dShape};
  }

}
