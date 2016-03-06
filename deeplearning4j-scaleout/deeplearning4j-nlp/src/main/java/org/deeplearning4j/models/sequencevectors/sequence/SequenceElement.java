package org.deeplearning4j.models.sequencevectors.sequence;

import com.google.common.util.concurrent.AtomicDouble;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 *  SequenceElement is basic building block for SequenceVectors. Any data sequence can be represented as ordered set of SequenceElements,
 *  and then one can learn distributed representation of each SequenceElement in this sequence using CBOW or SkipGram.
 *
 * @author raver119@gmail.com
 */
public abstract class SequenceElement implements Comparable<SequenceElement>, Serializable {

    private static final long serialVersionUID = 2223750736522624732L;

    protected AtomicDouble elementFrequency = new AtomicDouble(0);

    //used in comparison when building the huffman tree
    protected int index = -1;
    protected List<Integer> codes = new ArrayList<>();

    protected INDArray historicalGradient;
    protected List<Integer> points = new ArrayList<>();
    protected int codeLength = 0;

    // this var defines, if this token can't be truncated with minWordFrequency threshold
    @Getter @Setter protected boolean special;

    // this var defines that we have label here
    protected boolean isLabel;

    protected AdaGrad adaGrad;

    /*
            Reserved for Joint/Distributed vocabs mechanics
    */
    @Getter @Setter protected Long storageId;

    /**
     * This method should return string representation of this SequenceElement, so it can be used for
     *
     * @return
     */
    abstract public String getLabel();


    /**
     * Returns whether this element was defined as label, or no
     *
     * @return
     */
    public boolean isLabel() {
        return isLabel;
    }

    /**
     * This method specifies, whether this element should be treated as label for some sequence/document or not.
     *
     * @param isLabel
     */
    public void markAsLabel(boolean isLabel) {
        this.isLabel = isLabel;
    }

    /**
     * This method returns SequenceElement's frequency in current training corpus.
     *
     * @return
     */
    public double getElementFrequency() {
        return elementFrequency.get();
    }

    /**
     * This method sets frequency value for this element
     *
     * @param value
     */
    public void setElementFrequency(long value) {
        elementFrequency.set(value);
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
     * Sets Huffman tree points
     *
     * @param points
     */
    public void setPoints(int[] points) {
        this.points = new ArrayList<>();
        for (int i = 0; i < points.length; i++) {
            this.points.add(points[i]);
        }
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
            for(int i = 0; i < codeLength; i++)
                codes.add(0);
        }

        if(points.size() < codeLength) {
            for(int i = 0; i < codeLength; i++)
                points.add(0);
        }
    }


    /**
     * Returns gradient for this specific element, at specific position
     * @param index
     * @param g
     * @param lr
     * @return
     */
    public double getGradient(int index, double g, double lr) {
        if (adaGrad == null)
            adaGrad = new AdaGrad(1,getCodeLength(), lr);

        return adaGrad.getGradient(g, index, new int[]{1, getCodeLength()});
    }

    public void setHistoricalGradient(INDArray gradient) {
        if (adaGrad == null)
            adaGrad = new AdaGrad(1,getCodeLength(), 0.025);

        adaGrad.setHistoricalGradient(gradient);
    }

    public INDArray getHistoricalGradient() {
        if (adaGrad == null)
            adaGrad = new AdaGrad(1,getCodeLength(), 0.025);
        return adaGrad.getHistoricalGradient();
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
