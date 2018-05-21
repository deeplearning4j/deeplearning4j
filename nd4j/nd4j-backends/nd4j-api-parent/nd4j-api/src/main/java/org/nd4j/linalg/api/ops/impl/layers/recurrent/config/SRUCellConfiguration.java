package org.nd4j.linalg.api.ops.impl.layers.recurrent.config;

import lombok.Builder;
import lombok.Data;
import org.nd4j.autodiff.samediff.SDVariable;

@Data
@Builder
public class SRUCellConfiguration {
    /**
     *
     NDArray<T>* xt   = INPUT_VARIABLE(0);               // input [batchSize x inSize], batchSize - batch size, inSize - number of features
     NDArray<T>* ct_1 = INPUT_VARIABLE(1);               // previous cell state ct  [batchSize x inSize], that is at previous time step t-1
     NDArray<T>* w    = INPUT_VARIABLE(2);               // weights [inSize x 3*inSize]
     NDArray<T>* b    = INPUT_VARIABLE(3);               // biases [1 × 2*inSize]

     NDArray<T>* ht   = OUTPUT_VARIABLE(0);              // current cell output [batchSize x inSize], that is at current time step t
     NDArray<T>* ct   = OUTPUT_VARIABLE(1);              // current cell state  [batchSize x inSize], that is at current time step t

     */
    private SDVariable xt,ct_1,w,b,h1,ct;


    public SDVariable[] args() {
        return new SDVariable[] {xt,ct_1,w,b,h1,ct};
    }

}
