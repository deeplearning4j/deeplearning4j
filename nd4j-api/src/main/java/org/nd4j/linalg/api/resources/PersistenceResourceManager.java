package org.nd4j.linalg.api.resources;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

/**
 * Frees resources based
 *  on how long alive
 *  and whether the buffer is persistent or not.
 *
 *
 *
 *  @author Adam Gibson
 */
public class PersistenceResourceManager implements ResourceManager {
    private Map<String,INDArray> entries = Collections.synchronizedMap(new WeakHashMap<String,INDArray>());
    //weak hash maps are bad at multi threading, handle the traffic from other threads
    //and append bits at a time
    private Map<String,INDArray> recentAdds = new ConcurrentHashMap<>();
    private Map<String,Long> created = Collections.synchronizedMap(new HashMap<String,Long>());
    public final static long ALIVE_DURATION = 60;


    @Override
    public void register(INDArray arr) {
        recentAdds.put(arr.id(),arr);
        created.put(arr.id(),System.currentTimeMillis());
    }



    @Override
    public   void purge() {
        Set<String> remove = new HashSet<>();
        Set<String> keys = new HashSet<>(entries().keySet());
        for(String s : keys) {
            INDArray get = entries().get(s);
            if(shouldCollect(get)) {
                created.remove(s);
                get.data().destroy();
                remove.add(s);
            }
        }


        entries().putAll(recentAdds);
        recentAdds.clear();
    }

    private synchronized Map<String,INDArray> entries() {
        return entries;
    }


    @Override
    public boolean shouldCollect(INDArray collect) {
        long curr = System.currentTimeMillis();
        Long created = this.created.get(collect.id());
        if(created != null) {
            long diff = TimeUnit.MILLISECONDS.toSeconds(Math.abs(curr - created));
            return !collect.data().isPersist() && diff >= ALIVE_DURATION;
        }
        else
            return !collect.data().isPersist();

    }
}
