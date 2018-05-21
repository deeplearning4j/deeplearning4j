package org.deeplearning4j.clustering.randomprojection;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

@Data
public class RPHyperPlanes {
    private int dim;
    private INDArray wholeHyperPlane;

    public RPHyperPlanes(int dim) {
        this.dim = dim;
    }

    public INDArray getHyperPlaneAt(int depth) {
        if(wholeHyperPlane.isVector())
            return wholeHyperPlane;
        return wholeHyperPlane.slice(depth);
    }


    /**
     * Add a new random element to the hyper plane.
     */
    public void addRandomHyperPlane() {
        INDArray newPlane = Nd4j.randn(new int[] {1,dim});
        newPlane.divi(newPlane.normmaxNumber());
        if(wholeHyperPlane == null)
            wholeHyperPlane = newPlane;
        else {
            wholeHyperPlane = Nd4j.concat(0,wholeHyperPlane,newPlane);
        }
    }


}
