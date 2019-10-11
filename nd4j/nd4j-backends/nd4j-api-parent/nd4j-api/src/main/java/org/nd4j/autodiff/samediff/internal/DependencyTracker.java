package org.nd4j.autodiff.samediff.internal;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NonNull;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;

/**
 *
 * @param <T> For a dependency X -> Y, Y has type T
 * @param <D> For a dependency X -> Y, X has type D
 */
public class DependencyTracker<T, D> {

    private Queue<T> zeroDependencyQueue = new LinkedList<>();
    private Set<T> zeroDependenciesSet = new HashSet<>();           //Same content as queue, but set for O(1) contains
    private Map<T, Set<D>> dependencies = new HashMap<>();
    private Map<T, Set<Pair<D,D>>> orDependencies = new HashMap<>();

    private Map<D,D> aliases = new HashMap<>();                  //Key: Dependency; value: "real"/underlying dependency
    private Map<D,Set<D>> aliasesReverse = new HashMap<>();      //Key: real/underlying dependency; value: all values that alias this real one


    public void clear(){
        zeroDependencyQueue.clear();
        zeroDependenciesSet.clear();
        dependencies.clear();
        orDependencies.clear();
        aliases.clear();
    }

    /**
     * Check whether any dependencies x -> y exist
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

        //Check for aliases
        if(aliases.containsKey(x))
            x = aliases.get(x);

        dependencies.get(y).add(x);
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

        //Check for aliases
        if(aliases.containsKey(x))
            x = aliases.get(x);


        Set<D> s = dependencies.get(y);
        if(s != null) {
            s.remove(x);
        }

        Set<Pair<D,D>> s2 = orDependencies.get(y);
        if(s2 != null) {
            Iterator<Pair<D,D>> iter = s2.iterator();
            while(iter.hasNext()){
                Pair<D,D> p = iter.next();
                if(x.equals(p.getFirst()) || x.equals(p.getSecond())){
                    iter.remove();
                }
            }
        }

        //Add to the zero dependency queue/set
        if((s == null || s.isEmpty()) && (s2 == null || s2.isEmpty())){
            if(!zeroDependenciesSet.contains(y)){
                zeroDependenciesSet.add(y);
                zeroDependencyQueue.add(y);
                dependencies.remove(y);
                orDependencies.remove(y);
            }
        }
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

        //Check for aliases
        if(aliases.containsKey(x1))
            x1 = aliases.get(x1);
        if(aliases.containsKey(x2))
            x2 = aliases.get(x2);

        orDependencies.get(y).add(new Pair<>(x1, x2));
    }



    /**
     * Store the given value, as having no dependencies before it can be used/executed/made available
     * @param y The value to add
     */
    public void addZeroDependencyItem(T y){
        if(!zeroDependenciesSet.contains(y)){
            zeroDependenciesSet.add(y);
            zeroDependencyQueue.add(y);
        }
    }

    public boolean hasZeroDependencyItem(){
        return !zeroDependencyQueue.isEmpty();
    }

    public T removeZeroDependencyItem(){
        Preconditions.checkState(hasZeroDependencyItem(), "Not zero dependency items available (hasZeroDependencyItem() == false)");

        T ret = zeroDependencyQueue.remove();
        zeroDependenciesSet.remove(ret);

        return ret;
    }

    public List<T> removeAllZeroDependencyItems(){
        Preconditions.checkState(hasZeroDependencyItem(), "Not zero dependency items available (hasZeroDependencyItem() == false)");
        List<T> ret = new ArrayList<>(zeroDependencyQueue);
        zeroDependencyQueue.clear();
        zeroDependenciesSet.clear();

        return ret;
    }

    /**
     * Add an alias - "alias" is the same thing as "real"
     * @param real
     * @param alias
     */
    public void addAlias(@NonNull D real, @NonNull D alias){
        Preconditions.checkState(!real.equals(alias), "Cannot create an alias of itself: real=%s, alias=%s", real, alias);

        /*
        Handle transitive aliases.
        So:
        dt.addAlias(x, y);
        dt.addAlias(y, z);

        is equivalent to:

        dt.addAlias(x, y);
        dt.addAlias(x, z);

        Because y is an alias of x
         */


        while(aliases.containsKey(real)){
            real = aliases.get(real);
        }

        aliases.put(alias, real);
        if(!aliasesReverse.containsKey(real))
            aliasesReverse.put(real, new HashSet<D>());
        aliasesReverse.get(real).add(alias);
    }

    /**
     * Returns true if argument x is an alias for y
     * @param x
     * @return
     */
    public boolean isAlias(D x){
        return aliases.containsKey(x);
    }

    /**
     * If x is an alias of y, get y
     * @param x
     * @return
     */
    public D aliasGetUnderlying(D x){
        Preconditions.checkState(isAlias(x), "Argument is not registered as an alias: %s", x);
        return aliases.get(x);
    }

    public void removeAlias(@NonNull D alias){
        D underlying = aliases.remove(alias);
        if(underlying != null){
            aliasesReverse.get(underlying).remove(alias);
        }
    }
}
