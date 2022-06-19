package org.nd4j.autodiff.samediff.internal;


import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.function.Predicate;

public class DependencMapLinkedHash<K, V> implements IDependencyMap<K,V> {
    //IDependeeGroup will act as dummy interface and will be ignored

    private HashMap<K, HashSet<V>> map = new LinkedHashMap<K, HashSet<V>>();  
    @Override
    public void clear() {
        map.clear();
    }

    @Override
    public void add(K dependeeGroup, V element) {
      HashSet<V> s = map.get(dependeeGroup);
      if(s==null){
        s= new HashSet<V> ();
        map.put(dependeeGroup, s);
      }
       s.add(element);
    }

    @Override
    public Iterable<V> getDependantsForEach(K dependeeGroup) {
        return map.get(dependeeGroup);
    }

    @Override
    public Iterable<V> getDependantsForGroup(K dependeeGroup) {
        return map.get(dependeeGroup);
    }

    @Override
    public boolean containsAny(K dependeeGroup) {
        return map.containsKey(dependeeGroup);
    }

    @Override
    public boolean containsAnyForGroup(K dependeeGroup) {
        return map.containsKey(dependeeGroup);
    }

    @Override
    public boolean isEmpty() {
        return map.isEmpty();
    }

    @Override
    public void removeGroup(K dependeeGroup) {
        map.remove(dependeeGroup);
    }

    @Override
    public Iterable<V> removeGroupReturn(K dependeeGroup) {
        return map.remove(dependeeGroup);
    }

    @Override
    public void removeForEach(K dependeeGroup) {
          map.remove(dependeeGroup);
    }

    @Override
    public Iterable<V> removeForEachResult(K dependeeGroup) {
        return map.remove(dependeeGroup);
    }

    @Override
    public Iterable<V> removeGroupReturn(K dependeeGroup, Predicate<V> predicate) {
        HashSet<V> s= new HashSet<V> ();
        HashSet<V> ret = map.get(dependeeGroup);
        if(ret!=null){
            long prevSize = ret.size();
            for (V v : ret) {
                if(predicate.test(v)) s.add(v);
            }
            for (V v : s) {
                ret.remove(s);
            }
            //remove the key as well
            if(prevSize == s.size()){
                //remove the key
                //as we are testing containsAny using key
                map.remove(dependeeGroup);
            }
        }
        return s;
    }
    
}
