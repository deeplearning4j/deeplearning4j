package org.nd4j.jita.allocator.impl;

import lombok.*;
import org.nd4j.jita.allocator.concurrency.AtomicState;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.allocator.enums.AccessState;
import org.nd4j.jita.allocator.time.RateTimer;
import org.nd4j.jita.allocator.time.impl.BinaryTimer;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@NoArgsConstructor
public class NestedPoint {
    @Getter @Setter @NonNull private AllocationShape shape;
    @Getter @Setter @NonNull private AtomicState accessState;
    private AtomicLong accessTime;
    @Getter private RateTimer timerShort = new BinaryTimer(10, TimeUnit.SECONDS);
    @Getter private RateTimer timerLong = new BinaryTimer(60, TimeUnit.SECONDS);


    // by default memory is UNDEFINED, and depends on parent memory chunk for now
    @Getter @Setter private AllocationStatus nestedStatus = AllocationStatus.UNDEFINED;

    private AtomicLong counter = new AtomicLong(0);

    public NestedPoint(@NonNull AllocationShape shape) {
        this.shape = shape;
    }

    /**
     * Returns number of ticks for this point
     *
     * @return
     */
    public long getTicks() {
        return counter.get();
    }

    /**
     * Increments number of ticks by one
     */
    public void tick() {
        accessTime.set(System.nanoTime());
        this.counter.incrementAndGet();
    }

    public void tack() {
        // TODO: to be implemented
        // TODO: or not
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        NestedPoint that = (NestedPoint) o;

        return shape != null ? shape.equals(that.shape) : that.shape == null;

    }

    @Override
    public int hashCode() {
        return shape != null ? shape.hashCode() : 0;
    }
}
