package org.nd4j.linalg.api.ops.impl.layers.recurrent.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.samediff.SDVariable;

@Data
@Builder
public class SRUConfiguration {
    /**
     * NDArray<T>* input   = INPUT_VARIABLE(0);                // X, input 3d tensor [bS x K x N], N - number of time steps, bS - batch size, K - number of features
     NDArray<T>* weights = INPUT_VARIABLE(1);                // W, 2d tensor of weights [3K x K]
     NDArray<T>* bias    = INPUT_VARIABLE(2);                // B, row of biases with twice length [1 × 2*K]
     NDArray<T>* init    = INPUT_VARIABLE(3);                // C_{0}, 2d tensor of initial state [bS x K] at time t=0

     */
    private SDVariable inputs,weights,bias,init;

    public SDVariable[] args() {
        return new SDVariable[] {inputs,weights,bias,init};
    }
}
