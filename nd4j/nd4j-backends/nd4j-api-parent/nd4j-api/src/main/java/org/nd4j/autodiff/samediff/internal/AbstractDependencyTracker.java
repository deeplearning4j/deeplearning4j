package org.nd4j.autodiff.samediff.internal;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.function.Predicate;
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
public abstract class AbstractDependencyTracker<T, D> {
    private final Map<T, Set<D>> dependencies;
    private final Map<T, Set<Pair<D,D>>> orDependencies;
    private final Map<D, Set<T>> reverseDependencies = new HashMap<>();
    private final Map<D, Set<T>> reverseOrDependencies = new HashMap<>();
    private final Set<D> satisfiedDependencies = new HashSet<>();         //Mark the dependency as satisfied. If not in set: assumed to not be satisfied

    private final Set<T> allSatisfied;
    private final Queue<T> allSatisfiedQueue = new LinkedList<>();

    protected abstract Map<T, ?> newTMap();

    protected abstract Set<T> newTSet();
    protected AbstractDependencyTracker(){
        dependencies = (Map<T, Set<D>>) newTMap();
        orDependencies = (Map<T, Set<Pair<D,D>>>) newTMap();
        allSatisfied = newTSet();
    }

    public void clear(){
        dependencies.clear();
        orDependencies.clear();
        reverseDependencies.clear();
        reverseOrDependencies.clear();
        satisfiedDependencies.clear();
        allSatisfied.clear();
        allSatisfiedQueue.clear();
    }

    public boolean isEmpty(){
        return dependencies.isEmpty() && orDependencies.isEmpty() &&
                allSatisfiedQueue.isEmpty();
    }

    public boolean isSatisfied(@NonNull D x){
        return satisfiedDependencies.contains(x);
    }

    public void markSatisfied(@NonNull D x, boolean satisfied){
        if(satisfied){
            boolean alreadySatisfied = satisfiedDependencies.contains(x);

            if (!alreadySatisfied) {
                satisfiedDependencies.add(x);

                //Check if any Y's exist that have dependencies that are all satisfied, for X -> Y
                Set<T> s = reverseDependencies.get(x);
                Set<T> s2 = reverseOrDependencies.get(x);

                Set<T> set;
                if(s != null && s2 != null){
                    set = newTSet();
                    set.addAll(s);
                    set.addAll(s2);
                } else if(s != null){
                    set = s;
                } else if(s2 != null){
                    set = s2;
                } else {
                    log.info("No values depend on: {}", x);     //TODO Remove debugging
                    return;
                }

                for (T t : set) {
                    Set<D> required = dependencies.get(t);
                    Set<Pair<D,D>> requiredOr = orDependencies.get(t);
                    boolean allSatisfied = true;
                    if(required != null) {
                        for (D d : required) {
                            if (!isSatisfied(d)) {
                                allSatisfied = false;
                                break;
                            }
                        }
                    }
                    if(allSatisfied && requiredOr != null){
                        for(Pair<D,D> p : requiredOr){
                            if(!isSatisfied(p.getFirst()) && !isSatisfied(p.getSecond())){
                                allSatisfied = false;
                                break;
                            }
                        }
                    }

                    if (allSatisfied) {
                        if(!this.allSatisfied.contains(t)){
                            this.allSatisfied.add(t);
                            this.allSatisfiedQueue.add(t);
                        }
                    }
                }
            }

        } else {
            satisfiedDependencies.remove(x);
            throw new UnsupportedOperationException("Not yet implemented: Need to check all satisfied queue and update...");
        }
    }

    /**
     * Check whether any dependencies x -> y exist, for y
     * @param y
     * @return
     */
    public boolean hasDependency(@NonNull T y){
        Set<D> s1 = dependencies.get(y);
        if(s1 != null && !s1.isEmpty())
            return true;

        Set<Pair<D,D>> s2 = orDependencies.get(y);
        return s2 != null && !s2.isEmpty();
    }

    /**
     * Get all dependencies x, for x -> y, and (x1 or x2) -> y
     * @param y
     * @return
     */
    public DependencyList<T,D> getDependencies(@NonNull T y){
        Set<D> s1 = dependencies.get(y);
        Set<Pair<D,D>> s2 = orDependencies.get(y);

        List<D> l1 = (s1 == null ? null : new ArrayList<>(s1));
        List<Pair<D,D>> l2 = (s2 == null ? null : new ArrayList<>(s2));

        return new DependencyList<>(y, l1, l2);
    }

    /**
     * Add a dependency: y depends on x, as in x -> y
     * @param y
     * @param x The dependency that is required for Y
     */
    public void addDependency(@NonNull T y, @NonNull D x){
        if(!dependencies.containsKey(y))
            dependencies.put(y, new HashSet<D>());

        if(!reverseDependencies.containsKey(x))
            reverseDependencies.put(x, newTSet());

        dependencies.get(y).add(x);
        reverseDependencies.get(x).add(y);
    }


    /**
     * Remove a dependency (x -> y)
     *
     * @param y
     * @param x The dependency that is no longer required for Y
     */
    public void removeDependency(@NonNull T y, @NonNull D x){
        if(!dependencies.containsKey(y) && !orDependencies.containsKey(y))
            return;

        Set<D> s = dependencies.get(y);
        if(s != null) {
            s.remove(x);
            if(s.isEmpty())
                dependencies.remove(y);
        }

        Set<T> s2 = reverseDependencies.get(x);
        if(s2 != null){
            s2.remove(y);
            if(s2.isEmpty())
                reverseDependencies.remove(x);
        }


        Set<Pair<D,D>> s3 = orDependencies.get(y);
        if(s3 != null) {
            boolean removedReverse = false;
            Iterator<Pair<D,D>> iter = s3.iterator();
            while(iter.hasNext()){
                Pair<D,D> p = iter.next();
                if(x.equals(p.getFirst()) || x.equals(p.getSecond())){
                    iter.remove();

                    if(!removedReverse) {
                        Set<T> set1 = reverseOrDependencies.get(p.getFirst());
                        Set<T> set2 = reverseOrDependencies.get(p.getSecond());

                        set1.remove(y);
                        set2.remove(y);

                        if(set1.isEmpty())
                            reverseOrDependencies.remove(p.getFirst());
                        if(set2.isEmpty())
                            reverseOrDependencies.remove(p.getSecond());

                        removedReverse = true;
                    }
                }
            }
        }
        if(s3 != null && s3.isEmpty())
            orDependencies.remove(y);
    }

    /**
     * Add an "Or" dependency: y requires either x1 OR x2
     *
     * @param y
     * @param x1
     * @param x2
     */
    public void addOrDependency(@NonNull T y, @NonNull D x1, @NonNull D x2){
        if(!orDependencies.containsKey(y))
            orDependencies.put(y, new HashSet<Pair<D,D>>());

        if(!reverseOrDependencies.containsKey(x1))
            reverseOrDependencies.put(x1, newTSet());
        if(!reverseOrDependencies.containsKey(x2))
            reverseOrDependencies.put(x2, newTSet());

        orDependencies.get(y).add(new Pair<>(x1, x2));
        reverseOrDependencies.get(x1).add(y);
        reverseOrDependencies.get(x2).add(y);
    }


    public boolean hasNewAllSatisfied(){
        return !allSatisfiedQueue.isEmpty();
    }

    public T getNewAllSatisfied(){
        Preconditions.checkState(hasNewAllSatisfied(), "No new/unprocessed dependents that are all satisfied");
        return allSatisfiedQueue.remove();
    }

    public List<T> getNewAllSatisfiedList(){
        Preconditions.checkState(hasNewAllSatisfied(), "No new/unprocessed dependents that are all satisfied");
        List<T> ret = new ArrayList<>(allSatisfiedQueue);
        allSatisfiedQueue.clear();
        return ret;
    }

    public T getFirstNewAllSatisfiedMatching(@NonNull Predicate<T> predicate){
        Preconditions.checkState(hasNewAllSatisfied(), "No new/unprocessed dependents that are all satisfied");

        T t = allSatisfiedQueue.peek();
        if(predicate.test(t)){
            t = allSatisfiedQueue.remove();
            allSatisfied.remove(t);
            return t;
        }

        if(allSatisfiedQueue.size() > 1){
            Iterator<T> iter = allSatisfiedQueue.iterator();
            while(iter.hasNext()){
                t = iter.next();
                if(predicate.test(t)){
                    iter.remove();
                    allSatisfied.remove(t);
                    return t;
                }
            }
        }

        return null;    //None match predicate
    }

}
