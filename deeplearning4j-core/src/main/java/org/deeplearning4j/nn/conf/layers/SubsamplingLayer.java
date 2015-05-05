package org.deeplearning4j.nn.conf.layers;

/**
 * @author Adam Gibson
 */
public class SubsamplingLayer extends Layer {
    
    private static final long serialVersionUID = -7095644470333017030L;

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
        return "SubsamplingLayer{" +
                '}';
    }
}
