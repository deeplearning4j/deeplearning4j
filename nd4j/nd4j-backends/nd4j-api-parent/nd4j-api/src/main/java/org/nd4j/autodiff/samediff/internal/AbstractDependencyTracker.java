/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff.internal;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.function.Predicate;
import org.nd4j.common.primitives.Pair;

import java.util.*;

@Slf4j
public abstract class AbstractDependencyTracker<T, D> {
    @Getter
    private final IDependencyMap<T, D> dependencies;
    @Getter
    private final IDependencyMap<T, Pair<D, D>> orDependencies;
    @Getter
    private final Map<D, Set<T>> reverseDependencies = new LinkedHashMap<>();
    @Getter
    private final Map<D, Set<T>> reverseOrDependencies = new HashMap<>();
    @Getter
    private final Set<D> satisfiedDependencies = new LinkedHashSet<>();
    @Getter
    private final Set<T> allSatisfied;
    @Getter
    private final Queue<T> allSatisfiedQueue = new LinkedList<>();

    // Cross-frame dependency alias tracking for dependees (D type)
    private final Map<D, D> dependeeAliases = new HashMap<>();

    protected AbstractDependencyTracker() {
        dependencies = (IDependencyMap<T, D>) newTMap();
        orDependencies = (IDependencyMap<T, Pair<D, D>>) newTMap();
        allSatisfied = newTSet();
    }

    /**
     * @return A new map where the dependents (i.e., Y in "X -> Y") are the key
     */
    protected abstract IDependencyMap<T, ?> newTMap();

    /**
     * @return A new set where the dependents (i.e., Y in "X -> Y") are the key
     */
    protected abstract Set<T> newTSet();

    /**
     * @return A String representation of the dependent object
     */
    protected abstract String toStringT(T t);

    /**
     * @return A String representation of the dependee object
     */
    protected abstract String toStringD(D d);

    /**
     * Resolve cross-frame dependencies for control flow operations
     * This method should be called after Enter operations complete to establish
     * proper dependency links for Merge operations
     */
    public void resolveCrossFrameDependencies(Map<String, FrameIter> frameTransitions) {
        Map<D, D> newAliases = new HashMap<>();

        // For each frame transition, create appropriate aliases
        for (Map.Entry<String, FrameIter> entry : frameTransitions.entrySet()) {
            String varName = entry.getKey();
            FrameIter targetFrame = entry.getValue();

            // Find any existing dependencies that should be aliased to the new frame
            for (Map.Entry<D, Set<T>> dep : reverseDependencies.entrySet()) {
                D dependee = dep.getKey();

                // Check if this dependee matches the variable name and should be aliased
                String dependeeName = toStringD(dependee);
                if (dependeeName.contains(varName)) {
                    // Create a new dependee for the target frame
                    // This is a generic approach - specific implementations may need to override
                    try {
                        // Use reflection to create a new instance of the same type
                        D newDependee = createDependeeForFrame(dependee, varName, targetFrame);
                        if (newDependee != null) {
                            newAliases.put(dependee, newDependee);
                        }
                    } catch (Exception e) {
                        log.warn("Could not create cross-frame alias for {}: {}", dependeeName, e.getMessage());
                    }
                }
            }
        }

        if (!newAliases.isEmpty()) {
            batchCreateDependeeAliases(newAliases);
            log.debug("Created {} cross-frame dependency aliases", newAliases.size());
        }
    }

    /**
     * Create a new dependee for the specified frame
     * Subclasses should override this method to provide type-specific implementations
     */
    protected D createDependeeForFrame(D originalDependee, String varName, FrameIter targetFrame) {
        // Default implementation returns null - subclasses should override
        return null;
    }

    /**
     * Clear all internal state for the dependency tracker
     */
    public void clear() {
        dependencies.clear();
        orDependencies.clear();
        reverseDependencies.clear();
        reverseOrDependencies.clear();
        satisfiedDependencies.clear();
        allSatisfied.clear();
        allSatisfiedQueue.clear();
        dependeeAliases.clear();
    }

    /**
     * @return True if no dependencies have been defined
     */
    public boolean isEmpty() {
        return dependencies.isEmpty() && orDependencies.isEmpty() &&
                allSatisfiedQueue.isEmpty();
    }

    /**
     * Resolve dependee aliases - follow the chain to get the actual dependee
     */
    private D resolveDependeeAlias(D dependee) {
        D current = dependee;
        Set<D> visited = new HashSet<>();

        while (dependeeAliases.containsKey(current)) {
            if (visited.contains(current)) {
                log.warn("Circular dependee alias detected for: {}", toStringD(dependee));
                break;
            }
            visited.add(current);
            current = dependeeAliases.get(current);
        }

        return current;
    }

    /**
     * @return True if the dependency has been marked as satisfied using
     *         {@link #markSatisfied(Object, boolean)}
     */
    public boolean isSatisfied(@NonNull D x) {
        D resolved = resolveDependeeAlias(x);
        return satisfiedDependencies.contains(resolved);
    }

    /**
     * Mark the specified value as satisfied.
     * For example, if two dependencies have been previously added (X -> Y) and (X
     * -> A) then after the markSatisfied(X, true)
     * call, both of these dependencies are considered satisfied.
     *
     * @param x         Value to mark
     * @param satisfied Whether to mark as satisfied (true) or unsatisfied (false)
     */
    public void markSatisfied(@NonNull D x, boolean satisfied) {
        D resolved = resolveDependeeAlias(x);

        if (satisfied) {
            boolean alreadySatisfied = satisfiedDependencies.contains(resolved);

            if (!alreadySatisfied) {
                satisfiedDependencies.add(resolved);

                // Check if any Y's exist that have dependencies that are all satisfied, for X -> Y
                Set<T> s = reverseDependencies.get(resolved);
                Set<T> s2 = reverseOrDependencies.get(resolved);

                Set<T> set;
                if (s != null && s2 != null) {
                    set = newTSet();
                    set.addAll(s);
                    set.addAll(s2);
                } else if (s != null) {
                    set = s;
                } else if (s2 != null) {
                    set = s2;
                } else {
                    if (log.isTraceEnabled()) {
                        log.trace("No values depend on: {}", toStringD(resolved));
                    }
                    return;
                }

                for (T t : set) {
                    boolean allSatisfied = true;
                    Iterable<D> it = dependencies.getDependantsForEach(t);
                    if (it != null) {
                        for (D d : it) {
                            if (!isSatisfied(d)) {
                                allSatisfied = false;
                                break;
                            }
                        }
                    }

                    if (allSatisfied) {
                        Iterable<Pair<D, D>> itOr = orDependencies.getDependantsForEach(t);
                        if (itOr != null) {
                            for (Pair<D, D> p : itOr) {
                                if (!isSatisfied(p.getFirst()) && !isSatisfied(p.getSecond())) {
                                    allSatisfied = false;
                                    break;
                                }
                            }
                        }
                    }

                    if (allSatisfied && !this.allSatisfied.contains(t)) {
                        this.allSatisfied.add(t);
                        this.allSatisfiedQueue.add(t);
                    }
                }
            }

        } else {
            satisfiedDependencies.remove(resolved);
            if (!allSatisfied.isEmpty()) {

                Set<T> reverse = reverseDependencies.get(resolved);
                if (reverse != null) {
                    for (T y : reverse) {
                        if (allSatisfied.contains(y)) {
                            allSatisfied.remove(y);
                            allSatisfiedQueue.remove(y);
                        }
                    }
                }
                Set<T> orReverse = reverseOrDependencies.get(resolved);
                if (orReverse != null) {
                    for (T y : orReverse) {
                        if (allSatisfied.contains(y) && !isAllSatisfied(y)) {
                            allSatisfied.remove(y);
                            allSatisfiedQueue.remove(y);
                        }
                    }
                }
            }
        }
    }

    /**
     * Check whether any dependencies x -> y exist, for y (i.e., anything previously
     * added by {@link #addDependency(Object, Object)}
     * or {@link #addOrDependency(Object, Object, Object)}
     *
     * @param y Dependent to check
     * @return True if Y depends on any values
     */
    public boolean hasDependency(@NonNull T y) {
        return dependencies.containsAny(y) || orDependencies.containsAny(y);
    }

    /**
     * Get all dependencies x, for x -> y, and (x1 or x2) -> y
     *
     * @param y Dependent to get dependencies for
     * @return List of dependencies
     */
    public DependencyList<T, D> getDependencies(@NonNull T y) {
        Iterable<D> s1 = dependencies.getDependantsForEach(y);
        Iterable<Pair<D, D>> s2 = orDependencies.getDependantsForEach(y);

        return new DependencyList<>(y, s1, s2);
    }

    /**
     * Add a dependency: y depends on x, as in x -> y
     *
     * @param y The dependent
     * @param x The dependee that is required for Y
     */
    public void addDependency(@NonNull T y, @NonNull D x) {
        D resolved = resolveDependeeAlias(x);

        if (!reverseDependencies.containsKey(resolved))
            reverseDependencies.put(resolved, newTSet());

        dependencies.add(y, resolved);
        reverseDependencies.get(resolved).add(y);

        checkAndUpdateIfAllSatisfied(y);
    }

    protected void checkAndUpdateIfAllSatisfied(@NonNull T y) {
        boolean allSat = isAllSatisfied(y);
        if (allSat) {
            // Case where "x is satisfied" happened before x->y added
            if (!allSatisfied.contains(y)) {
                allSatisfied.add(y);
                allSatisfiedQueue.add(y);
            }
        } else if (allSatisfied.contains(y)) {
            if (!allSatisfiedQueue.contains(y)) {
                StringBuilder sb = new StringBuilder();
                sb.append("Dependent object \"").append(toStringT(y))
                        .append("\" was previously processed after all dependencies")
                        .append(" were marked satisfied, but is now additional dependencies have been added.\n");
                Iterable<D> dl = dependencies.getDependantsForEach(y);
                if (dl != null) {
                    sb.append("Dependencies:\n");
                    for (D d : dl) {
                        sb.append(d).append(" - ").append(isSatisfied(d) ? "Satisfied" : "Not satisfied").append("\n");
                    }
                }
                Iterable<Pair<D, D>> dlOr = orDependencies.getDependantsForEach(y);
                if (dlOr != null) {
                    sb.append("Or dependencies:\n");
                    for (Pair<D, D> p : dlOr) {
                        sb.append(p).append(" - satisfied=(").append(isSatisfied(p.getFirst())).append(",")
                                .append(isSatisfied(p.getSecond())).append(")");
                    }
                }

                allSatisfiedQueue.add(y);
                log.warn(sb.toString());
            }

            // Not satisfied, but is in the queue -> needs to be removed
            allSatisfied.remove(y);
            allSatisfiedQueue.remove(y);
        }
    }

    protected boolean isAllSatisfied(@NonNull T y) {
        Iterable<D> set1 = dependencies.getDependantsForEach(y);

        boolean retVal = true;
        if (set1 != null) {
            for (D d : set1) {
                retVal = isSatisfied(d);
                if (!retVal)
                    break;
            }
        }
        if (retVal) {
            Iterable<Pair<D, D>> set2 = orDependencies.getDependantsForEach(y);
            if (set2 != null) {
                for (Pair<D, D> p : set2) {
                    retVal = isSatisfied(p.getFirst()) || isSatisfied(p.getSecond());
                    if (!retVal)
                        break;
                }
            }
        }

        return retVal;
    }

    /**
     * Remove a dependency (x -> y)
     *
     * @param y The dependent that currently requires X
     * @param x The dependee that is no longer required for Y
     */
    public void removeDependency(@NonNull T y, @NonNull D x) {
        D resolved = resolveDependeeAlias(x);

        dependencies.removeGroupReturn(y, t -> t.equals(resolved));

        Set<T> s2 = reverseDependencies.get(resolved);
        if (s2 != null) {
            s2.remove(y);
            if (s2.isEmpty())
                reverseDependencies.remove(resolved);
        }

        Iterable<Pair<D, D>> s3 = orDependencies.removeGroupReturn(y, t -> {
            D first = resolveDependeeAlias(t.getFirst());
            D second = resolveDependeeAlias(t.getSecond());
            return resolved.equals(first) || resolved.equals(second);
        });

        if (s3 != null) {
            boolean removedReverse = false;
            for (Pair<D, D> p : s3) {
                if (!removedReverse) {
                    D first = resolveDependeeAlias(p.getFirst());
                    D second = resolveDependeeAlias(p.getSecond());

                    Set<T> set1 = reverseOrDependencies.get(first);
                    Set<T> set2 = reverseOrDependencies.get(second);

                    if (set1 != null) {
                        set1.remove(y);
                        if (set1.isEmpty()) reverseOrDependencies.remove(first);
                    }
                    if (set2 != null) {
                        set2.remove(y);
                        if (set2.isEmpty()) reverseOrDependencies.remove(second);
                    }

                    removedReverse = true;
                }
            }
        }
    }

    /**
     * Add an "Or" dependency: Y requires either x1 OR x2 - i.e., (x1 or x2) -> Y<br>
     * If either x1 or x2 (or both) are marked satisfied via
     * {@link #markSatisfied(Object, boolean)} then the
     * dependency is considered satisfied
     *
     * @param y  Dependent
     * @param x1 Dependee 1
     * @param x2 Dependee 2
     */
    public void addOrDependency(@NonNull T y, @NonNull D x1, @NonNull D x2) {
        D resolved1 = resolveDependeeAlias(x1);
        D resolved2 = resolveDependeeAlias(x2);

        if (!reverseOrDependencies.containsKey(resolved1))
            reverseOrDependencies.put(resolved1, newTSet());
        if (!reverseOrDependencies.containsKey(resolved2))
            reverseOrDependencies.put(resolved2, newTSet());

        orDependencies.add(y, new Pair<>(resolved1, resolved2));
        reverseOrDependencies.get(resolved1).add(y);
        reverseOrDependencies.get(resolved2).add(y);

        checkAndUpdateIfAllSatisfied(y);
    }

    /**
     * @return True if there are any new/unprocessed "all satisfied dependents" (Ys
     *         in X->Y)
     */
    public boolean hasNewAllSatisfied() {
        return !allSatisfiedQueue.isEmpty();
    }

    /**
     * Returns the next new dependent (Y in X->Y) that has all dependees (Xs) marked
     * as satisfied via {@link #markSatisfied(Object, boolean)}
     * Throws an exception if {@link #hasNewAllSatisfied()} returns false.<br>
     * Note that once a value has been retrieved from here, no new dependencies of
     * the form (X -> Y) can be added for this value;
     * the value is considered "processed" at this point.
     *
     * @return The next new "all satisfied dependent"
     */
    public T getNewAllSatisfied() {
        Preconditions.checkState(hasNewAllSatisfied(), "No new/unprocessed dependents that are all satisfied");
        return allSatisfiedQueue.remove();
    }

    /**
     * @return As per {@link #getNewAllSatisfied()} but returns all values
     */
    public List<T> getNewAllSatisfiedList() {
        Preconditions.checkState(hasNewAllSatisfied(), "No new/unprocessed dependents that are all satisfied");
        List<T> ret = new ArrayList<>(allSatisfiedQueue);
        allSatisfiedQueue.clear();
        return ret;
    }

    /**
     * As per {@link #getNewAllSatisfied()} but instead of returning the first
     * dependee, it returns the first that matches
     * the provided predicate. If no value matches the predicate, null is returned
     *
     * @param predicate Predicate for checking
     * @return The first value matching the predicate, or null if no values match
     *         the predicate
     */
    public T getFirstNewAllSatisfiedMatching(@NonNull Predicate<T> predicate) {
        Preconditions.checkState(hasNewAllSatisfied(), "No new/unprocessed dependents that are all satisfied");

        T t = allSatisfiedQueue.peek();
        if (predicate.test(t)) {
            t = allSatisfiedQueue.remove();
            allSatisfied.remove(t);
            return t;
        }

        if (allSatisfiedQueue.size() > 1) {
            Iterator<T> iter = allSatisfiedQueue.iterator();
            while (iter.hasNext()) {
                t = iter.next();
                if (predicate.test(t)) {
                    iter.remove();
                    allSatisfied.remove(t);
                    return t;
                }
            }
        }

        return null; // None match predicate
    }

    /**
     * Create an alias mapping from oldDependee to newDependee
     * This allows operations to find dependencies that have moved between frames
     */
    public void createDependeeAlias(@NonNull D oldDependee, @NonNull D newDependee) {
        dependeeAliases.put(oldDependee, newDependee);

        // If the old dependee was satisfied, mark the new one as satisfied
        if (satisfiedDependencies.contains(oldDependee)) {
            satisfiedDependencies.remove(oldDependee);
            satisfiedDependencies.add(newDependee);
        }

        log.debug("Created dependee alias: {} -> {}", toStringD(oldDependee), toStringD(newDependee));
    }

    /**
     * Batch create dependee aliases
     */
    public void batchCreateDependeeAliases(@NonNull Map<D, D> aliasMapping) {
        for (Map.Entry<D, D> entry : aliasMapping.entrySet()) {
            dependeeAliases.put(entry.getKey(), entry.getValue());
        }

        // Handle satisfaction transfer
        for (Map.Entry<D, D> entry : aliasMapping.entrySet()) {
            if (satisfiedDependencies.contains(entry.getKey())) {
                satisfiedDependencies.remove(entry.getKey());
                satisfiedDependencies.add(entry.getValue());
            }
        }

        // Re-evaluate dependencies that might now be satisfied
        reevaluateAllSatisfied();

        log.debug("Batch created {} dependee aliases", aliasMapping.size());
    }



    /**
     * Force re-evaluation of all satisfied dependencies
     */
    public void reevaluateAllSatisfied() {
        Set<T> previouslySatisfied = new HashSet<>(allSatisfied);
        allSatisfied.clear();
        allSatisfiedQueue.clear();

        // Re-check all previously satisfied dependents
        for (T dependent : previouslySatisfied) {
            if (isAllSatisfied(dependent)) {
                allSatisfied.add(dependent);
                allSatisfiedQueue.add(dependent);
            }
        }

        log.debug("Re-evaluated satisfied dependencies: {} -> {}",
                previouslySatisfied.size(), allSatisfied.size());
    }
}