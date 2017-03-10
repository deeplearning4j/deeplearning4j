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

  protected double[] data;
  protected int[] shape;
  protected int[] stride;
  protected char order;

  public NDArrayHelper(INDArray array) {
    this.data = NDArrayHelper.toFlattened(array);
    this.shape = array.shape();
    this.stride = array.stride();
    this.order = 'c';
  }

  public static double[] toFlattened(INDArray array) {
    return array.dup('c').data().asDouble();
  }

}
