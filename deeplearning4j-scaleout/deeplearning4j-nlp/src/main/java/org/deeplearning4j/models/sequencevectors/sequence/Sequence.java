package org.deeplearning4j.models.sequencevectors.sequence;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.io.Serializable;
import java.util.*;

/**
 * Sequence for SequenceVectors is defined as limited set of SequenceElements. It can also contain label, if you're going to learn Sequence features as well.
 *
 * @author raver119@gmail.com
 */
public class Sequence<T extends SequenceElement> implements Serializable {

    private static final long serialVersionUID = 2223750736522624732L;

    protected List<T> elements = new ArrayList<>();

    // elements map needed to speedup searches againt elements in sequence
    protected Map<String, T> elementsMap = new LinkedHashMap<>();

    // each document can have multiple labels
    protected List<T> labels = new ArrayList<>();

    protected T label;

    @Getter @Setter protected int sequenceId;

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
    public synchronized void addElement(@NonNull T element) {
        this.elementsMap.put(element.getLabel(), element);
        this.elements.add(element);
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
        return elementsMap.get(label);
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
     * Returns all labels for this sequence
     *
     * @return
     */
    public List<T> getSequenceLabels() {
        return labels;
    }

    /**
     * Sets sequence labels
     * @param labels
     */
    public void setSequenceLabels(List<T> labels) {
        this.labels = labels;
    }

    /**
     * Set sequence label
     *
     * @param label
     */
    public void setSequenceLabel(@NonNull T label) {
        this.label = label;
        if (!labels.contains(label)) labels.add(label);
    }

    /**
     *  Adds sequence label. In this case sequence will have multiple labels
     *
     * @param label
     */
    public void addSequenceLabel(@NonNull T label) {
        this.labels.add(label);
        if (this.label == null) this.label = label;
    }

    @Override
    public String toString() {
        return "Sequence{" +
                " labels=" + labels +
                ", elements=" + elements +
                '}';
    }
}
