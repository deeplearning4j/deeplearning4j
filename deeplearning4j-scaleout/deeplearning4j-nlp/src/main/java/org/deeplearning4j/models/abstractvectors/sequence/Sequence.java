package org.deeplearning4j.models.abstractvectors.sequence;

import lombok.Getter;
import lombok.Setter;
import org.apache.commons.collections.map.HashedMap;

import java.util.*;

/**
 * Sequence for AbstractVectors is defined as limited set of SequenceElements. It can also contain label, if you're going to learn Sequence features as well.
 *
 * @author raver119@gmail.com
 */
public class Sequence<T extends SequenceElement> {
    protected Map<String, T> elements = new HashMap<String, T>();
    protected T label;

    public Sequence() {

    }

    public Sequence(Collection<T> set) {
        for (T element : set) {
            this.elements.put(element.getLabel(), element);
        }
    }

    public List<String> asLabels() {
        List<String> labels = new ArrayList<>();
        for(T element: getElements()) {
            labels.add(element.getLabel());
        }
        return labels;
    }

    public T getElementByLabel(String label) {
        return elements.get(label);
    }

    public Collection<T> getElements() {
        return Collections.unmodifiableCollection(elements.values());
    }
}
