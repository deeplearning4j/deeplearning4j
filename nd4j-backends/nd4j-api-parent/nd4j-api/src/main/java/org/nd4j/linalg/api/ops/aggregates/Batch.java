package org.nd4j.linalg.api.ops.aggregates;

import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.List;

/**
 * DO NOT EVER USE THIS CLASS, IT'S ONLY FOR DEVELOPMENT PURPOSES
 *
 * @author raver119@gmail.com
 */
@Deprecated
public class Batch {
    @Getter private List<Aggregate> aggregates = new ArrayList<>();
    @Getter private final long threadId;

    public Batch() {
        threadId = Thread.currentThread().getId();
    }

    public void enqueueAggregate(Aggregate op) {
        aggregates.add(op);
    }

    public int size() {
        return aggregates.size();
    }
}
