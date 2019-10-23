package org.nd4j.autodiff.samediff.internal;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.function.Predicate;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;

/**
 * Object dependency tracker.
 * <br>
 * Dependency are denoted by: X -> Y, which means "Y depends on X"<br>
 * In this implementation:<br>
 * - Dependencies may be satisfied, or not satisfied<br>
 * - The implementation tracks when the dependency for an object Y are fully satisfied. This occurs when:<br>
 * 1. No dependencies X->Y exist<br>
 * 2. All dependencies of the form X->Y have been marked as satisfied, via markSatisfied(x)<br>
 * - When a dependency is satisfied, any dependent (Ys) are checked to see if all their dependencies are satisfied<br>
 * - If a dependent has all dependencies satisfied, it is added to the "new all satisfied" queue for processing,
 * which can be accessed via {@link #hasNewAllSatisfied()}, {@link #getNewAllSatisfied()} and {@link #getNewAllSatisfiedList()}<br>
 * <br>
 * Note: Two types of dependencies exist<br>
 * 1. Standard dependencies - i.e., "Y depends on X"<br>
 * 2. "Or" dependencies - i.e., "Y depends on (A or B)".<br>
 * For Or dependencies of the form "(A or B) -> Y", Y will be marked as "all dependencies satisfied" if either A or B is marked as satisfied.
 *
 * @param <T> For a dependency X -> Y, Y has type T
 * @param <D> For a dependency X -> Y, X has type D
 */
@Slf4j
public abstract class AbstractDependencyTracker<T, D> {
    @Getter
    private final Map<T, Set<D>> dependencies;                              //Key: the dependent. Value: all things that the key depends on
    @Getter
    private final Map<T, Set<Pair<D, D>>> orDependencies;                    //Key: the dependent. Value: the set of OR dependencies
    private final Map<D, Set<T>> reverseDependencies = new HashMap<>();     //Key: the dependee. Value: The set of all dependents that depend on this value
    private final Map<D, Set<T>> reverseOrDependencies = new HashMap<>();
    private final Set<D> satisfiedDependencies = new HashSet<>();           //Mark the dependency as satisfied. If not in set: assumed to not be satisfied

    private final Set<T> allSatisfied;                                      //Set of all dependent values (Ys) that have all dependencies satisfied
    private final Queue<T> allSatisfiedQueue = new LinkedList<>();          //Queue for *new* "all satisfied" values. Values are removed using the "new all satisfied" methods


    protected AbstractDependencyTracker() {
        dependencies = (Map<T, Set<D>>) newTMap();
        orDependencies = (Map<T, Set<Pair<D, D>>>) newTMap();
        allSatisfied = newTSet();
    }

    /**
     * @return A new map where the dependents (i.e., Y in "X -> Y") are the key
     */
    protected abstract Map<T, ?> newTMap();

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
    }

    /**
     * @return True if no dependencies have been defined
     */
    public boolean isEmpty() {
        return dependencies.isEmpty() && orDependencies.isEmpty() &&
                allSatisfiedQueue.isEmpty();
    }

    /**
     * @return True if the dependency has been marked as satisfied using {@link #markSatisfied(Object, boolean)}
     */
    public boolean isSatisfied(@NonNull D x) {
        return satisfiedDependencies.contains(x);
    }

    /**
     * Mark the specified value as satisfied.
     * For example, if two dependencies have been previously added (X -> Y) and (X -> A) then after the markSatisfied(X, true)
     * call, both of these dependencies are considered satisfied.
     *
     * @param x         Value to mark
     * @param satisfied Whether to mark as satisfied (true) or unsatisfied (false)
     */
    public void markSatisfied(@NonNull D x, boolean satisfied) {
        if (satisfied) {
            boolean alreadySatisfied = satisfiedDependencies.contains(x);

            if (!alreadySatisfied) {
                satisfiedDependencies.add(x);

                //Check if any Y's exist that have dependencies that are all satisfied, for X -> Y
                Set<T> s = reverseDependencies.get(x);
                Set<T> s2 = reverseOrDependencies.get(x);

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
                        log.trace("No values depend on: {}", toStringD(x));
                    }
                    return;
                }

                for (T t : set) {
                    Set<D> required = dependencies.get(t);
                    Set<Pair<D, D>> requiredOr = orDependencies.get(t);
                    boolean allSatisfied = true;
                    if (required != null) {
                        for (D d : required) {
                            if (!isSatisfied(d)) {
                                allSatisfied = false;
                                break;
                            }
                        }
                    }
                    if (allSatisfied && requiredOr != null) {
                        for (Pair<D, D> p : requiredOr) {
                            if (!isSatisfied(p.getFirst()) && !isSatisfied(p.getSecond())) {
                                allSatisfied = false;
                                break;
                            }
                        }
                    }

                    if (allSatisfied) {
                        if (!this.allSatisfied.contains(t)) {
                            this.allSatisfied.add(t);
                            this.allSatisfiedQueue.add(t);
                        }
                    }
                }
            }

        } else {
            satisfiedDependencies.remove(x);
            if (!allSatisfied.isEmpty()) {

                Set<T> reverse = reverseDependencies.get(x);
                if (reverse != null) {
                    for (T y : reverse) {
                        if (allSatisfied.contains(y)) {
                            allSatisfied.remove(y);
                            allSatisfiedQueue.remove(y);
                        }
                    }
                }
                Set<T> orReverse = reverseOrDependencies.get(x);
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
     * Check whether any dependencies x -> y exist, for y (i.e., anything previously added by {@link #addDependency(Object, Object)}
     * or {@link #addOrDependency(Object, Object, Object)}
     *
     * @param y Dependent to check
     * @return True if Y depends on any values
     */
    public boolean hasDependency(@NonNull T y) {
        Set<D> s1 = dependencies.get(y);
        if (s1 != null && !s1.isEmpty())
            return true;

        Set<Pair<D, D>> s2 = orDependencies.get(y);
        return s2 != null && !s2.isEmpty();
    }

    /**
     * Get all dependencies x, for x -> y, and (x1 or x2) -> y
     *
     * @param y Dependent to get dependencies for
     * @return List of dependencies
     */
    public DependencyList<T, D> getDependencies(@NonNull T y) {
        Set<D> s1 = dependencies.get(y);
        Set<Pair<D, D>> s2 = orDependencies.get(y);

        List<D> l1 = (s1 == null ? null : new ArrayList<>(s1));
        List<Pair<D, D>> l2 = (s2 == null ? null : new ArrayList<>(s2));

        return new DependencyList<>(y, l1, l2);
    }

    /**
     * Add a dependency: y depends on x, as in x -> y
     *
     * @param y The dependent
     * @param x The dependee that is required for Y
     */
    public void addDependency(@NonNull T y, @NonNull D x) {
        if (!dependencies.containsKey(y))
            dependencies.put(y, new HashSet<D>());

        if (!reverseDependencies.containsKey(x))
            reverseDependencies.put(x, newTSet());

        dependencies.get(y).add(x);
        reverseDependencies.get(x).add(y);

        checkAndUpdateIfAllSatisfied(y);
    }

    protected void checkAndUpdateIfAllSatisfied(@NonNull T y) {
        boolean allSat = isAllSatisfied(y);
        if (allSat) {
            //Case where "x is satisfied" happened before x->y added
            if (!allSatisfied.contains(y)) {
                allSatisfied.add(y);
                allSatisfiedQueue.add(y);
            }
        } else if (allSatisfied.contains(y)) {
            if (!allSatisfiedQueue.contains(y)) {
                StringBuilder sb = new StringBuilder();
                sb.append("Dependent object \"").append(toStringT(y)).append("\" was previously processed after all dependencies")
                        .append(" were marked satisfied, but is now additional dependencies have been added.\n");
                DependencyList<T, D> dl = getDependencies(y);
                if (dl.getDependencies() != null) {
                    sb.append("Dependencies:\n");
                    for (D d : dl.getDependencies()) {
                        sb.append(d).append(" - ").append(isSatisfied(d) ? "Satisfied" : "Not satisfied").append("\n");
                    }
                }
                if (dl.getOrDependencies() != null) {
                    sb.append("Or dependencies:\n");
                    for (Pair<D, D> p : dl.getOrDependencies()) {
                        sb.append(p).append(" - satisfied=(").append(isSatisfied(p.getFirst())).append(",").append(isSatisfied(p.getSecond())).append(")");
                    }
                }
                throw new IllegalStateException(sb.toString());
            }

            //Not satisfied, but is in the queue -> needs to be removed
            allSatisfied.remove(y);
            allSatisfiedQueue.remove(y);
        }
    }

    protected boolean isAllSatisfied(@NonNull T y) {
        Set<D> set1 = dependencies.get(y);

        boolean allSatisfied = true;
        if (set1 != null) {
            for (D d : set1) {
                allSatisfied = isSatisfied(d);
                if (!allSatisfied)
                    break;
            }
        }
        if (allSatisfied) {
            Set<Pair<D, D>> set2 = orDependencies.get(y);
            if (set2 != null) {
                for (Pair<D, D> p : set2) {
                    allSatisfied = isSatisfied(p.getFirst()) || isSatisfied(p.getSecond());
                    if (!allSatisfied)
                        break;
                }
            }
        }
        return allSatisfied;
    }


    /**
     * Remove a dependency (x -> y)
     *
     * @param y The dependent that currently requires X
     * @param x The dependee that is no longer required for Y
     */
    public void removeDependency(@NonNull T y, @NonNull D x) {
        if (!dependencies.containsKey(y) && !orDependencies.containsKey(y))
            return;

        Set<D> s = dependencies.get(y);
        if (s != null) {
            s.remove(x);
            if (s.isEmpty())
                dependencies.remove(y);
        }

        Set<T> s2 = reverseDependencies.get(x);
        if (s2 != null) {
            s2.remove(y);
            if (s2.isEmpty())
                reverseDependencies.remove(x);
        }


        Set<Pair<D, D>> s3 = orDependencies.get(y);
        if (s3 != null) {
            boolean removedReverse = false;
            Iterator<Pair<D, D>> iter = s3.iterator();
            while (iter.hasNext()) {
                Pair<D, D> p = iter.next();
                if (x.equals(p.getFirst()) || x.equals(p.getSecond())) {
                    iter.remove();

                    if (!removedReverse) {
                        Set<T> set1 = reverseOrDependencies.get(p.getFirst());
                        Set<T> set2 = reverseOrDependencies.get(p.getSecond());

                        set1.remove(y);
                        set2.remove(y);

                        if (set1.isEmpty())
                            reverseOrDependencies.remove(p.getFirst());
                        if (set2.isEmpty())
                            reverseOrDependencies.remove(p.getSecond());

                        removedReverse = true;
                    }
                }
            }
        }
        if (s3 != null && s3.isEmpty())
            orDependencies.remove(y);
    }

    /**
     * Add an "Or" dependency: Y requires either x1 OR x2 - i.e., (x1 or x2) -> Y<br>
     * If either x1 or x2 (or both) are marked satisfied via {@link #markSatisfied(Object, boolean)} then the
     * dependency is considered satisfied
     *
     * @param y  Dependent
     * @param x1 Dependee 1
     * @param x2 Dependee 2
     */
    public void addOrDependency(@NonNull T y, @NonNull D x1, @NonNull D x2) {
        if (!orDependencies.containsKey(y))
            orDependencies.put(y, new HashSet<Pair<D, D>>());

        if (!reverseOrDependencies.containsKey(x1))
            reverseOrDependencies.put(x1, newTSet());
        if (!reverseOrDependencies.containsKey(x2))
            reverseOrDependencies.put(x2, newTSet());

        orDependencies.get(y).add(new Pair<>(x1, x2));
        reverseOrDependencies.get(x1).add(y);
        reverseOrDependencies.get(x2).add(y);

        checkAndUpdateIfAllSatisfied(y);
    }

    /**
     * @return True if there are any new/unprocessed "all satisfied dependents" (Ys in X->Y)
     */
    public boolean hasNewAllSatisfied() {
        return !allSatisfiedQueue.isEmpty();
    }

    /**
     * Returns the next new dependent (Y in X->Y) that has all dependees (Xs) marked as satisfied via {@link #markSatisfied(Object, boolean)}
     * Throws an exception if {@link #hasNewAllSatisfied()} returns false.<br>
     * Note that once a value has been retrieved from here, no new dependencies of the form (X -> Y) can be added for this value;
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
     * As per {@link #getNewAllSatisfied()} but instead of returning the first dependee, it returns the first that matches
     * the provided predicate. If no value matches the predicate, null is returned
     *
     * @param predicate Predicate gor checking
     * @return The first value matching the predicate, or null if no values match the predicate
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

        return null;    //None match predicate
    }
}
