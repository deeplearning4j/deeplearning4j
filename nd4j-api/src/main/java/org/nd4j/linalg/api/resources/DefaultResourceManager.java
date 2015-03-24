package org.nd4j.linalg.api.resources;

import com.google.common.collect.MapMaker;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;

/**
 * Default resource manager
 * @author Adam Gibson
 */
public class DefaultResourceManager implements ResourceManager {
    private Map<String,DataBuffer> entries = new MapMaker().weakValues().concurrencyLevel(8).makeMap();
    private Map<String,Long> created = Collections.synchronizedMap(new HashMap<String, Long>());

    public DefaultResourceManager() {
        ClassPathResource r = new ClassPathResource(NATIVE_PROPERTIES);
        if(!r.exists()) {
            maxAllocated.set(2048);
        }
        else {
            try {
                InputStream is = r.getInputStream();
                Properties props = new Properties();
                props.load(is);
                for(String s : props.stringPropertyNames())
                    System.setProperty(s,props.getProperty(s));
                maxAllocated.set(Long.parseLong(System.getProperty(MAX_ALLOCATED,"2048")));
                memoryRatio.set(Double.parseDouble(System.getProperty(MEMORY_RATIO,"0.9")));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

        }
    }

    @Override
    public double memoryRatio() {
        return memoryRatio.get();
    }

    @Override
    public void decrementCurrentAllocatedMemory(long decrement) {
        currentAllocated.set(currentAllocated.get() - decrement);
    }

    @Override
    public void incrementCurrentAllocatedMemory(long add) {
        currentAllocated.getAndAdd(add);
    }

    @Override
    public void setCurrentAllocated(long currentAllocated) {
        ResourceManager.currentAllocated.set(currentAllocated);
    }

    @Override
    public long maxAllocated() {
        return maxAllocated.get();
    }

    @Override
    public long currentAllocated() {
        return currentAllocated.get();
    }

    @Override
    public void register(INDArray arr) {
        entries.put(arr.id(),arr.data());
        created.put(arr.id(),System.currentTimeMillis());

    }

    @Override
    public void purge() {
       double ratio = Double.valueOf(currentAllocated()) / Double.valueOf(maxAllocated());
        if(ratio >= memoryRatio.get()) {
            for(String s : entries.keySet()) {
                if(!entries.get(s).isPersist() && entries.get(s).references().isEmpty())
                    entries.get(s).destroy();
            }

            System.runFinalization();

        }

    }

    @Override
    public boolean shouldCollect(INDArray collect) {
        return !collect.data().isPersist() &&  collect.data().references().isEmpty();

    }


}
