package org.nd4j.autodiff.samediff.internal;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;

/**
 * Object dependency tracker, using object identity (not object equality)
 *
 * Dependency are denoted by: X -> Y, which means "object Y depends on X"
 * In this implementation:<br>
 * - Dependencies may be satisfied, or not satisfied<br>
 * - The implementation tracks when the dependency for an object Y are fully satisfied. This occurs when:<br>
 *     1. No dependencies X->Y exist<br>
 *     2. All dependencies of the form X->Y have been marked as satisfied, via markSatisfied(x)<br>
 * - When a dependency is satisfied, any dependent (Ys) are checked to see if all their dependencies are satisfied<br>
 * - If a dependent has all dependencies satisfied, it is added to the "new all satisfied" queue for processing,
 *   which can be accessed via {@link #hasNewAllSatisfied()}, {@link #getNewAllSatisfied()} and {@link #getNewAllSatisfiedList()}
 *
 *
 * @param <T> For a dependency X -> Y, Y has type T
 * @param <D> For a dependency X -> Y, X has type D
 */
@Slf4j
public class IdentityDependencyTracker<T, D> extends AbstractDependencyTracker<T,D> {

    @Override
    protected Map<T, ?> newTMap() {
        return new IdentityHashMap<>();
    }

    @Override
    protected Set<T> newTSet() {
        return Collections.newSetFromMap(new IdentityHashMap<T, Boolean>());
    }
}
