package org.nd4j.autodiff.samediff.internal;

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

    private Map<T,T> dependentAliases = new HashMap<>();                  //Key: dependent; value: "real"/underlying dependent
    private Map<T,Set<T>> dependentAliasesReverse = new HashMap<>();      //Key: real/underlying dependent; value: all values that alias this real one

    private Map<D,D> dependeeAliases = new HashMap<>();                  //Key: Dependee; value: "real"/underlying dependee
    private Map<D,Set<D>> dependeeAliasesReverse = new HashMap<>();      //Key: real/underlying dependee; value: all values that alias this real one


    public void clear(){
        zeroDependencyQueue.clear();
        zeroDependenciesSet.clear();
        dependencies.clear();
        orDependencies.clear();
        dependeeAliases.clear();
    }

    public boolean isEmpty(){
        return zeroDependenciesSet.isEmpty() && zeroDependencyQueue.isEmpty() &&
                dependencies.isEmpty() && orDependencies.isEmpty() &&
                dependentAliases.isEmpty() && dependentAliasesReverse.isEmpty() &&
                dependeeAliases.isEmpty() && dependeeAliasesReverse.isEmpty();
    }

    /**
     * Check whether any dependencies x -> y exist, for y
     * @param y
     * @return
     */
    public boolean hasDependency(@NonNull T y){
        //Check if y is an alias, and get the 'real'/underlying value if so
        if(dependentAliases.containsKey(y))
            y = dependentAliases.get(y);

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
        //Check if y is an alias, and get the 'real'/underlying value if so
        if(dependentAliases.containsKey(y))
            y = dependentAliases.get(y);

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
        //Check for aliases
        if(dependentAliases.containsKey(y))
            y = dependentAliases.get(y);
        if(dependeeAliases.containsKey(x))
            x = dependeeAliases.get(x);

        if(!dependencies.containsKey(y))
            dependencies.put(y, new HashSet<D>());

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
        if(dependentAliases.containsKey(y))
            y = dependentAliases.get(y);
        if(dependeeAliases.containsKey(x))
            x = dependeeAliases.get(x);


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
            }
            dependencies.remove(y);
            orDependencies.remove(y);
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
        if(dependentAliases.containsKey(y))
            y = dependentAliases.get(y);
        if(dependeeAliases.containsKey(x1))
            x1 = dependeeAliases.get(x1);
        if(dependeeAliases.containsKey(x2))
            x2 = dependeeAliases.get(x2);

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
     * For dependencies of the form D -> T, add a dependee alias where D1 == D2, where D1 (the alias) should be treated
     * the same as D2 (the "real" instance)
     * @param real
     * @param alias
     */
    public void addDependeeAlias(@NonNull D real, @NonNull D alias){
        Preconditions.checkState(!real.equals(alias), "Cannot create an alias of itself: real=%s, alias=%s", real, alias);

        /*
        Handle transitive aliases.
        So:
        dt.addDependeeAlias(a, b);
        dt.addDependeeAlias(b, c);

        is equivalent to:

        dt.addAlias(a, b);
        dt.addAlias(a, c);

        Because b is an alias of a
         */


        while(dependeeAliases.containsKey(real)){
            real = dependeeAliases.get(real);
        }

        dependeeAliases.put(alias, real);
        if(!dependeeAliasesReverse.containsKey(real))
            dependeeAliasesReverse.put(real, new HashSet<D>());
        dependeeAliasesReverse.get(real).add(alias);
    }

    /**
     * Returns true if argument x is an alias for y
     * @param x
     * @return
     */
    public boolean isDependeeAlias(D x){
        return dependeeAliases.containsKey(x);
    }

    /**
     * If x is an alias of y, get y
     * @param x
     * @return
     */
    public D dependeeAliasGetUnderlying(D x){
        Preconditions.checkState(isDependeeAlias(x), "Argument is not registered as an alias: %s", x);
        return dependeeAliases.get(x);
    }

    public void removeDependeeAlias(@NonNull D alias){
        D underlying = dependeeAliases.remove(alias);
        if(underlying != null){
            Set<D> s = dependeeAliasesReverse.get(underlying);
            s.remove(alias);
            if(s.isEmpty()){
                dependeeAliasesReverse.remove(underlying);
            }
        }
    }










    /**
     * For dependencies of the form D -> T, add a dependent alias where T1 == T2, where T1 (the alias) should be treated
     * the same as T2 (the "real" instance)
     * @param real
     * @param alias
     */
    public void addDependentAlias(@NonNull T real, @NonNull T alias){
        Preconditions.checkState(!real.equals(alias), "Cannot create a self-referential alias (an alias of dependent that is itself): real=%s, alias=%s", real, alias);

        /*
        Handle transitive aliases.
        So:
        dt.addDependentAlias(x, y);
        dt.addDependentAlias(y, z);

        is equivalent to:

        dt.addDependentAlias(x, y);
        dt.addDependentAlias(x, z);

        Because y is an alias of x
         */


        while(dependentAliases.containsKey(real)){
            real = dependentAliases.get(real);
        }

        dependentAliases.put(alias, real);
        if(!dependentAliasesReverse.containsKey(real))
            dependentAliasesReverse.put(real, new HashSet<T>());
        dependentAliasesReverse.get(real).add(alias);
    }

    /**
     * Returns true if argument x is an alias for y
     * @param x
     * @return
     */
    public boolean isDependentAlias(T x){
        return dependentAliases.containsKey(x);
    }

    /**
     * If x is an alias of y, get y
     * @param x
     * @return
     */
    public T dependentAliasGetUnderlying(T x){
        Preconditions.checkState(isDependentAlias(x), "Argument is not registered as an alias: %s", x);
        return dependentAliases.get(x);
    }

    public void removeDependentAlias(@NonNull T alias){
        T underlying = dependentAliases.remove(alias);
        if(underlying != null){
            Set<T> s = dependentAliasesReverse.get(underlying);
            s.remove(alias);
            if(s.isEmpty()){
                dependentAliasesReverse.remove(underlying);
            }
        }
    }
}
