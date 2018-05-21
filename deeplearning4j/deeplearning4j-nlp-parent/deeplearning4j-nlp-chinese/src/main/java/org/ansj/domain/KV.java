package org.ansj.domain;

public class KV<K, V> {

    private K k;

    private V v;

    private KV(K k, V v) {
        this.k = k;
        this.v = v;
    }

    public static <K, V> KV<K, V> with(K k, V v) {
        return new KV<>(k, v);
    }

    public void setK(K k) {
        this.k = k;
    }

    public void setV(V v) {
        this.v = v;
    }

    public K getK() {
        return k;
    }

    public V getV() {
        return v;
    }
}
