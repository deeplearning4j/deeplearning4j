package org.ansj.library;

import org.ansj.dic.PathToStream;
import org.ansj.domain.KV;
import org.ansj.util.MyStaticValue;
import org.nlpcn.commons.lang.tire.domain.Forest;
import org.nlpcn.commons.lang.tire.domain.Value;
import org.nlpcn.commons.lang.tire.library.Library;
import org.nlpcn.commons.lang.util.IOUtil;
import org.nlpcn.commons.lang.util.StringUtil;
import org.nlpcn.commons.lang.util.logging.Log;

import java.io.BufferedReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class AmbiguityLibrary {

    private static final Log LOG = MyStaticValue.getLog(AmbiguityLibrary.class);

    // 同义词典
    private static final Map<String, KV<String, Forest>> AMBIGUITY = new HashMap<>();

    public static final String DEFAULT = "ambiguity";

    static {
        for (Entry<String, String> entry : MyStaticValue.ENV.entrySet()) {
            if (entry.getKey().startsWith(DEFAULT)) {
                put(entry.getKey(), entry.getValue());
            }
        }
        putIfAbsent(DEFAULT, "library/ambiguity.dic");
    }

    /**
     * 获取系统默认词典
     * 
     * @return
     */
    public static Forest get() {
        if (!AMBIGUITY.containsKey(DEFAULT)) {
            return null;
        }
        return get(DEFAULT);
    }

    /**
     * 根据key获取
     * 
     */
    public static Forest get(String key) {

        KV<String, Forest> kv = AMBIGUITY.get(key);

        if (kv == null) {
            if (MyStaticValue.ENV.containsKey(key)) {
                putIfAbsent(key, MyStaticValue.ENV.get(key));
                return get(key);
            }

            LOG.warn("crf " + key + " not found in config ");
            return null;
        }

        Forest sw = kv.getV();
        if (sw == null) {
            try {
                sw = init(key, kv, false);
            } catch (Exception e) {
            }
        }
        return sw;
    }

    /**
     * 加载
     * 
     * @return
     */
    private static synchronized Forest init(String key, KV<String, Forest> kv, boolean reload) {
        Forest forest = kv.getV();
        if (forest != null) {
            if (reload) {
                forest.clear();
            } else {
                return forest;
            }
        } else {
            forest = new Forest();
        }
        try (BufferedReader br = IOUtil.getReader(PathToStream.stream(kv.getK()), "utf-8")) {
            String temp;
            LOG.debug("begin init ambiguity");
            long start = System.currentTimeMillis();
            while ((temp = br.readLine()) != null) {
                if (StringUtil.isNotBlank(temp)) {
                    temp = StringUtil.trim(temp);
                    String[] split = temp.split("\t");
                    StringBuilder sb = new StringBuilder();
                    if (split.length % 2 != 0) {
                        LOG.error("init ambiguity  error in line :" + temp + " format err !");
                        continue;
                    }
                    for (int i = 0; i < split.length; i += 2) {
                        sb.append(split[i]);
                    }
                    forest.addBranch(sb.toString(), split);
                }
            }
            LOG.info("load dic use time:" + (System.currentTimeMillis() - start) + " path is : " + kv.getK());
            kv.setV(forest);
            return forest;
        } catch (Exception e) {
            LOG.error("Init ambiguity library error :" + e.getMessage() + ", path: " + kv.getK());
            AMBIGUITY.remove(key);
            return null;
        }
    }

    /**
     * 插入到树中呀
     * 
     * @param key
     * @param split
     * @return
     */
    public static void insert(String key, String... split) {
        Forest forest = get(key);
        StringBuilder sb = new StringBuilder();
        if (split.length % 2 != 0) {
            LOG.error("init ambiguity  error in line :" + Arrays.toString(split) + " format err !");
            return;
        }
        for (int i = 0; i < split.length; i += 2) {
            sb.append(split[i]);
        }
        forest.addBranch(sb.toString(), split);
    }

    /**
     * 插入到树种
     * 
     * @param key
     * @param value
     */
    public static void insert(String key, Value value) {
        Forest forest = get(key);
        Library.insertWord(forest, value);
    }

    /**
     * 动态添加
     * 
     * @param dicDefault
     * @param dicDefault2
     * @param dic2
     */
    public static void put(String key, String path) {
        put(key, path, null);
    }

    public static void put(String key, String path, Forest value) {
        AMBIGUITY.put(key, KV.with(path, value));
        MyStaticValue.ENV.put(key, path);
    }

    /**
     * 删除一个key
     * 
     * @param key
     * @return
     */
    public static KV<String, Forest> remove(String key) {
        KV<String, Forest> kv = AMBIGUITY.get(key);
        if (kv != null && kv.getV() != null) {
            kv.getV().clear();
        }
        MyStaticValue.ENV.remove(key);
        return AMBIGUITY.remove(key);
    }

    /**
     * 刷新一个,将值设置为null
     * 
     * @param key
     * @return
     */
    public static void reload(String key) {
        if (!MyStaticValue.ENV.containsKey(key)) { //如果变量中不存在直接删掉这个key不解释了
            remove(key);
        }

        putIfAbsent(key, MyStaticValue.ENV.get(key));

        KV<String, Forest> kv = AMBIGUITY.get(key);

        init(key, kv, true);
    }

    public static Set<String> keys() {
        return AMBIGUITY.keySet();
    }

    public static void putIfAbsent(String key, String path) {
        if (!AMBIGUITY.containsKey(key)) {
            AMBIGUITY.put(key, KV.with(path, (Forest) null));
        }
    }

}
