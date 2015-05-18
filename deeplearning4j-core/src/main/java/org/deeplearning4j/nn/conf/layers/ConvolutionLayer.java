package org.deeplearning4j.nn.conf.layers;

/**
 * @author Adam Gibson
 */
public class ConvolutionLayer extends Layer {

    /**
     * Convolution type: max avg or sum
     */
    public enum ConvolutionType {
        MAX,AVG,SUM,NONE
    }

    private static final long serialVersionUID = 3073633667258683720L;

    @Override
    public int hashCode() {
        return 0;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        return true;
    }

    public String toString() {
        return "ConvolutionLayer{" +
                '}';
    }
}
