package org.deeplearning4j.models.glove.count;

import lombok.Data;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * Simple POJO holding pairs of elements and their respective weights, used in GloVe -> CoOccurrence
 *
 * @author raver119@gmail.com
 */
@Data
public class CoOccurrenceWeight<T extends SequenceElement> {
    private T element1;
    private T element2;
    private double weight;

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        CoOccurrenceWeight<?> that = (CoOccurrenceWeight<?>) o;

        if (element1 != null ? !element1.equals(that.element1) : that.element1 != null)
            return false;
        return element2 != null ? element2.equals(that.element2) : that.element2 == null;

    }

    @Override
    public int hashCode() {
        int result = element1 != null ? element1.hashCode() : 0;
        result = 31 * result + (element2 != null ? element2.hashCode() : 0);
        return result;
    }
}
