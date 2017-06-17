package org.apdplat.word.vector;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * LRU (Least Recently Used) 算法的Java实现
 * @param <K>
 * @param <V>
 * @author 杨尚川
 */
public class LRUCache<K, V> extends LinkedHashMap<K, V> {
    //缓存大小
    private int cacheSize;

    public LRUCache(int cacheSize) {
        //第三个参数true是关键
        super(10, 0.75f, true);
        this.cacheSize = cacheSize;
    }

    /**
     * 缓存是否已满的判断
     * @param eldest
     * @return
     */
    @Override
    protected boolean removeEldestEntry(Map.Entry<K, V> eldest) {
        boolean r = size() > cacheSize;
        if(r){
            System.out.println("清除缓存："+eldest.getKey());
        }
        return r;
    }

    public static void main(String[] args) {
        LRUCache<String, String> cache = new LRUCache<>(5);
        cache.put("1", "1");
        cache.put("2", "2");
        cache.put("3", "3");
        cache.put("4", "4");
        cache.put("5", "5");

        System.out.println("初始：");
        cache.keySet().forEach(k -> System.out.println(k));
        System.out.println("访问2：");
        cache.get("2");
        cache.keySet().forEach(k -> System.out.println(k));
        System.out.println("访问2、3：");
        cache.get("2");
        cache.get("3");
        cache.keySet().forEach(k -> System.out.println(k));
        System.out.println("增加数据6、7：");
        cache.put("6", "6");
        cache.put("7", "7");
        cache.keySet().forEach(k -> System.out.println(k));
    }
}