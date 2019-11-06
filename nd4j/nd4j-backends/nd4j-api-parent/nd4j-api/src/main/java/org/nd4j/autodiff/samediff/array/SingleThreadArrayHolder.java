package org.nd4j.autodiff.samediff.array;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class SingleThreadArrayHolder implements ArrayHolder {

    private final Map<String, INDArray> map = new HashMap<>();

    @Override
    public boolean hasArray(@NonNull String name) {
        return map.containsKey(name);
    }

    @Override
    public INDArray getArray(@NonNull String name) {
        return map.get(name);
    }

    @Override
    public void setArray(@NonNull String name, @NonNull INDArray array) {
        map.put(name, array);
    }

    @Override
    public INDArray removeArray(@NonNull String name) {
        return map.remove(name);
    }

    @Override
    public int size() {
        return map.size();
    }

    @Override
    public void initFrom(ArrayHolder arrayHolder) {
        map.clear();
        Collection<String> names = arrayHolder.arrayNames();
        for(String n : names){
            map.put(n, arrayHolder.getArray(n));
        }
    }

    @Override
    public Collection<String> arrayNames() {
        return Collections.unmodifiableCollection(map.keySet());
    }

    @Override
    public void rename(String from, String to) {
        INDArray arr = map.remove(from);
        map.put(to, arr);
    }
}
