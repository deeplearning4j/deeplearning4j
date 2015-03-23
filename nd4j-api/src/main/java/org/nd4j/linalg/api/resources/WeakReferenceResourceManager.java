package org.nd4j.linalg.api.resources;

import com.google.common.collect.MapMaker;
import com.google.common.collect.Sets;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Weak reference resource manager
 * @author Adam Gibson
 */
public class WeakReferenceResourceManager implements ResourceManager {
    private Map<String,DataBuffer> entries = new MapMaker().weakValues().concurrencyLevel(8).makeMap();
    private Map<String,Long> created = Collections.synchronizedMap(new HashMap<String, Long>());

    @Override
    public void register(INDArray arr) {
        entries.put(arr.id(),arr.data());
        created.put(arr.id(),System.currentTimeMillis());

    }

    @Override
    public void purge() {
        for(String s : entries.keySet()) {
            if(!entries.get(s).isPersist() &&  entries.get(s).references().isEmpty())
                entries.get(s).destroy();
            else {
                Long get = created.get(s);
                long curr = System.currentTimeMillis();
                //delete any ndarrays a minute or more old
                if(!entries.get(s).isPersist()) {
                    long diff = TimeUnit.MILLISECONDS.toSeconds(Math.abs(get - curr));
                    if(diff >= 60) {
                        entries.get(s).destroy();
                    }
                }
            }
        }

        //purge stale keys
        Set<String> diff = Sets.difference(entries.keySet(),created.keySet());
        for(String item : diff)
            created.remove(item);
    }

    @Override
    public boolean shouldCollect(INDArray collect) {
        return !collect.data().isPersist() &&  collect.data().references().isEmpty();

    }


}
