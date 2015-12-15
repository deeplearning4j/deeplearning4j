package org.deeplearning4j.models.abstractvectors.sequence;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.apache.commons.collections.map.HashedMap;

import java.io.Serializable;
import java.util.*;

/**
 * Sequence for AbstractVectors is defined as limited set of SequenceElements. It can also contain label, if you're going to learn Sequence features as well.
 *
 * @author raver119@gmail.com
 */
public class Sequence<T extends SequenceElement> implements Serializable {

    private static final long serialVersionUID = 2223750736522624731L;

    protected List<T> elements = new ArrayList<>();
    protected T label;

    /**
     * Creates new empty sequence
     *
     */
    public Sequence() {

    }

    /**
     * Creates new sequence from collection of elements
     *
     * @param set
     */
    public Sequence(@NonNull Collection<T> set) {
        this();
        addElements(set);
    }

    /**
     * Adds single element to sequence
     *
     * @param element
     */
    public void addElement(@NonNull T element) {
        this.elements.add( element);
    }

    /**
     * Adds collection of elements to the sequence
     *
     * @param set
     */
    public void addElements(Collection<T> set) {
        for (T element : set) {
            addElement(element);
        }
    }

    /**
     * Returns this sequence as list of labels
     * @return
     */
    public List<String> asLabels() {
        List<String> labels = new ArrayList<>();
        for(T element: getElements()) {
            labels.add(element.getLabel());
        }
        return labels;
    }

    /**
     * Returns single element out of this sequence by its label
     *
     * @param label
     * @return
     */
    public T getElementByLabel(@NonNull String label) {
        for (T element: elements) {
            if (element.getLabel().equals(label)) return element;
        }
        return null;
    }

    /**
     * Returns ordered list of elements from this sequence
     *
     * @return
     */
    public List<T> getElements() {
        return Collections.unmodifiableList(elements);
    }

    /**
     * Returns label for this sequence
     *
     * @return label for this sequence, null if label was not defined
     */
    public T getSequenceLabel() {
        return label;
    }

    /**
     * Set sequence label
     *
     * @param label
     */
    public void setSequenceLabel(@NonNull T label) {
       this.label = label;
    }

    @Override
    public String toString() {
        return "Sequence{" +
                " label=" + label +
                ", elements=" + elements +
                '}';
    }
}
