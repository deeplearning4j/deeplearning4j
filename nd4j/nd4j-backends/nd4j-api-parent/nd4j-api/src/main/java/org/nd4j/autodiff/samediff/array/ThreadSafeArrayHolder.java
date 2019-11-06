package org.nd4j.autodiff.samediff.array;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.DeviceLocalNDArray;

import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * An {@link ArrayHolder} that uses the thread safe {@link DeviceLocalNDArray} internally
 *
 * @author Alex Black
 */
public class ThreadSafeArrayHolder implements ArrayHolder {

    private final Map<String, DeviceLocalNDArray> map = new ConcurrentHashMap<>();
    private final boolean lazyInit;

    /**
     * @param lazyInit If true: use lazy initialization for {@link DeviceLocalNDArray}
     */
    public ThreadSafeArrayHolder(boolean lazyInit) {
        this.lazyInit = lazyInit;
    }

    @Override
    public boolean hasArray(@NonNull String name) {
        return map.containsKey(name);
    }

    @Override
    public INDArray getArray(@NonNull String name) {
        return map.get(name).get();
    }

    @Override
    public void setArray(@NonNull String name, @NonNull INDArray array) {
        if (array.isView())
            array = array.dup();    //Device local doesn't support views
        if (!map.containsKey(name)) {
            DeviceLocalNDArray dla = new DeviceLocalNDArray(array, lazyInit);
            map.put(name, dla);
        } else {
            DeviceLocalNDArray dla = map.get(name);
            dla.update(array);
        }
    }

    @Override
    public INDArray removeArray(@NonNull String name) {
        DeviceLocalNDArray arr = map.remove(name);
        if (arr == null)
            return null;
        return arr.get();
    }

    @Override
    public int size() {
        return map.size();
    }

    @Override
    public void initFrom(ArrayHolder arrayHolder) {
        map.clear();
        Collection<String> names = arrayHolder.arrayNames();
        for (String n : names) {
            setArray(n, arrayHolder.getArray(n));
        }
    }

    @Override
    public Collection<String> arrayNames() {
        return Collections.unmodifiableCollection(map.keySet());
    }

    @Override
    public void rename(@NonNull String from, @NonNull String to) {
        DeviceLocalNDArray dl = map.remove(from);
        map.put(to, dl);
    }
}
