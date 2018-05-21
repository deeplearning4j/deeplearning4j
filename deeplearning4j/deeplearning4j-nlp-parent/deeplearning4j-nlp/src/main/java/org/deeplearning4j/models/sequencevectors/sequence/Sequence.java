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

    private static final long serialVersionUID = 2223750736522624735L;

    protected List<T> elements = new ArrayList<>();

    // elements map needed to speedup searches against elements in sequence
    protected Map<String, T> elementsMap = new LinkedHashMap<>();

    // each document can have multiple labels
    protected List<T> labels = new ArrayList<>();

    protected T label;

    protected int hash = 0;
    protected boolean hashCached = false;

    @Getter
    @Setter
    protected int sequenceId;

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
        hashCached = false;
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
        for (T element : getElements()) {
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
     * Returns an ordered unmodifiable list of elements from this sequence
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
        if (!labels.contains(label))
            labels.add(label);
    }

    /**
     *  Adds sequence label. In this case sequence will have multiple labels
     *
     * @param label
     */
    public void addSequenceLabel(@NonNull T label) {
        this.labels.add(label);
        if (this.label == null)
            this.label = label;
    }

    /**
     * Checks, if sequence is empty
     *
     * @return TRUE if empty, FALSE otherwise
     */
    public boolean isEmpty() {
        return elements.isEmpty();
    }

    /**
     * This method returns number of elements in this sequence
     *
     * @return
     */
    public int size() {
        return elements.size();
    }

    /**
     * This method returns  sequence element by index
     *
     * @param index
     * @return
     */
    public T getElementByIndex(int index) {
        return elements.get(index);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        Sequence<?> sequence = (Sequence<?>) o;

        return elements != null ? elements.equals(sequence.elements) : sequence.elements == null;

    }

    @Override
    public int hashCode() {
        if (hashCached)
            return hash;

        for (T element : elements) {
            hash += 31 * element.hashCode();
        }

        hashCached = true;

        return hash;
    }
}
