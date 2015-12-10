package org.deeplearning4j.models.abstractvectors.sequence;

import com.google.common.util.concurrent.AtomicDouble;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 *  SequenceElement is basic building block for AbstractVectors. Any data sequence can be represented as ordered set of SequenceElements,
 *  and then one can learn distributed representation of each SequenceElement in this sequence using CBOW or SkipGram.
 *
 * @author raver119@gmail.com
 */
public abstract class SequenceElement implements Comparable<SequenceElement> {
    protected AtomicDouble elementFrequency = new AtomicDouble(0);

    //used in comparison when building the huffman tree
    protected int index = -1;
    protected List<Integer> codes = new ArrayList<>();

    @Getter @Setter protected INDArray historicalGradient;
    protected List<Integer> points = new ArrayList<>();
    protected int codeLength = 0;
    @Getter @Setter protected int special;

    /*
            Used for Joint/Distributed vocabs mechanics
    */
    @Getter @Setter protected Long storageId;

    /**
     * This method should return string representation of this SequenceElement, so it can be used for
     *
     * @return
     */
    abstract public String getLabel();


    /**
     * This method returns SequenceElement's frequency in current training corpus.
     *
     * @return
     */
    public double getElementFrequency() {
        return elementFrequency.get();
    }

    /**
     * Increases element frequency counter by 1
     */
    public void incrementElementFrequency() {
        increaseElementFrequency(1);
    }

    /**
     * Increases element frequency counter by argument
     *
     * @param by
     */
    public void increaseElementFrequency(int by) {
        elementFrequency.getAndAdd(by);
    }

    /**
     * Equals method override should be properly implemented for any extended class, otherwise it will be based on label equality
     *
     * @param object
     * @return
     */
     public boolean equals(Object object) {
         if (this == object) return true;
         if (object == null) return false;
         if (!(object instanceof SequenceElement)) return false;

         return this.getLabel().equals(((SequenceElement) object).getLabel());
     }

    /**
     *  Returns index in Huffman tree
     *
     * @return index >= 0, if tree was built, -1 otherwise
     */
    public int getIndex() {
        return index;
    }

    /**
     * Sets index in Huffman tree
     *
     * @param index
     */
    public void setIndex(int index) {
        this.index = index;
    }

    /**
     * Returns Huffman tree codes
     * @return
     */
    public List<Integer> getCodes() {
        return codes;
    }

    /**
     * Sets Huffman tree codes
     * @param codes
     */
    public void setCodes(List<Integer> codes) {
        this.codes = codes;
    }

    /**
     * Returns Huffman tree points
     *
     * @return
     */
    public List<Integer> getPoints() {
        return points;
    }

    /**
     * Sets Huffman tree points
     *
     * @param points
     */
    public void setPoints(List<Integer> points) {
        this.points = points;
    }

    /**
     * Returns Huffman code length.
     *
     * Please note: maximum vocabulary/tree size depends on code length
     *
     * @return
     */
    public int getCodeLength() {
        return codeLength;
    }

    /**
     * This method fills codes and points up to codeLength
     *
     * @param codeLength
     */
    public void setCodeLength(int codeLength) {
        this.codeLength = codeLength;
        if(codes.size() < codeLength) {
            for(int i = codes.size(); i < codeLength; i++)
                codes.add(0);
        }

        if(points.size() < codeLength) {
            for(int i = codes.size(); i < codeLength; i++)
                points.add(0);
        }
    }

    /*
        TODO: fix this. AdaGrad here should be unified with the rest of dl4j
     */
    public double getGradient(int index, double g) {
        if(historicalGradient == null) {
            historicalGradient = Nd4j.zeros(getCodes().size());
        }

        double pow =  Math.pow(g,2);
        historicalGradient.putScalar(index, historicalGradient.getDouble(index) + pow);
        double sqrt =  FastMath.sqrt(historicalGradient.getDouble(index));
        double abs = FastMath.abs(g) / (sqrt + 1e-6f);
        double ret = abs * 1e-1f;
        return ret;
    }

    /**
     * hashCode method override should be properly implemented for any extended class, otherwise it will be based on label hashCode
     *
     * @return hashCode for this SequenceElement
     */
    public int hashCode() {
        if (this.getLabel() == null) throw new IllegalStateException("Label should not be null");
        return this.getLabel().hashCode();
    }

    @Override
    public int compareTo(SequenceElement o) {
        return Double.compare(elementFrequency.get(), o.elementFrequency.get());
    }

    @Override
    public String toString() {
        return "SequenceElement: {label: '"+ this.getLabel() +"'," +
                                                                  " freq: '"+ elementFrequency.get()+"'," +
                                                                    "index: '"+this.index+"'}";
    }

    /**
     *
     * @return
     */
    public abstract String toJSON();
}
