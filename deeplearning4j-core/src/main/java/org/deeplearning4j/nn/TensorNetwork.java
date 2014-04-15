package org.deeplearning4j.nn;


/**
 * An extension of the neural network, retains a lot of the same properties but W is now a tensor.
 * @author Adam Gibson
 */
public interface TensorNetwork extends NeuralNetwork {

    public Tensor getWTensor();

    public  void setW(Tensor w);


}
