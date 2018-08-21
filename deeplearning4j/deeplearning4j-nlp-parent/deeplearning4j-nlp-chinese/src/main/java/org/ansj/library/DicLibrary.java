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
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.BufferedReader;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class DicLibrary {

    private static final Log LOG = LogFactory.getLog();

    public static final String DEFAULT = "dic";

    public static final String DEFAULT_NATURE = "userDefine";

    public static final Integer DEFAULT_FREQ = 1000;

    public static final String DEFAULT_FREQ_STR = "1000";

    // 用户自定义词典
    private static final Map<String, KV<String, Forest>> DIC = new HashMap<>();

    static {
        for (Entry<String, String> entry : MyStaticValue.ENV.entrySet()) {
            if (entry.getKey().startsWith(DEFAULT)) {
                put(entry.getKey(), entry.getValue());
            }
        }
        putIfAbsent(DEFAULT, "library/default.dic");

        Forest forest = get();
        if (forest == null) {
            put(DEFAULT, DEFAULT, new Forest());
        }

    }

    /**
     * 关键词增加
     *
     * @param keyword 所要增加的关键词
     * @param nature 关键词的词性
     * @param freq 关键词的词频
     */
    public static void insert(String key, String keyword, String nature, int freq) {
        Forest dic = get(key);
        String[] paramers = new String[2];
        paramers[0] = nature;
        paramers[1] = String.valueOf(freq);
        Value value = new Value(keyword, paramers);
        Library.insertWord(dic, value);
    }

    /**
     * 增加关键词
     *
     * @param keyword
     */
    public static void insert(String key, String keyword) {

        insert(key, keyword, DEFAULT_NATURE, DEFAULT_FREQ);
    }

    /**
     * 删除关键词
     */
    public static void delete(String key, String word) {

        Forest dic = get(key);
        if (dic != null) {
            Library.removeWord(dic, word);
        }
    }

    /**
     * 将用户自定义词典清空
     */
    public static void clear(String key) {
        get(key).clear();
    }

    public static Forest get() {
        if (!DIC.containsKey(DEFAULT)) {
            return null;
        }
        return get(DEFAULT);
    }

    /**
     * 根据模型名称获取crf模型
     * 
     * @param modelName
     * @return
     */
    public static Forest get(String key) {

        KV<String, Forest> kv = DIC.get(key);

        if (kv == null) {
            if (MyStaticValue.ENV.containsKey(key)) {
                putIfAbsent(key, MyStaticValue.ENV.get(key));
                return get(key);
            }
            LOG.warn("dic " + key + " not found in config ");
            return null;
        }
        Forest forest = kv.getV();
        if (forest == null) {
            forest = init(key, kv, false);
        }
        return forest;

    }

    /**
     * 根据keys获取词典集合
     * 
     * @param keys
     * @return
     */
    public static Forest[] gets(String... keys) {
        Forest[] forests = new Forest[keys.length];
        for (int i = 0; i < forests.length; i++) {
            forests[i] = get(keys[i]);
        }
        return forests;
    }

    /**
     * 根据keys获取词典集合
     * 
     * @param keys
     * @return
     */
    public static Forest[] gets(Collection<String> keys) {
        return gets(keys.toArray(new String[keys.size()]));
    }

    /**
     * 用户自定义词典加载
     * 
     * @param key
     * @param path
     * @return
     */

    private synchronized static Forest init(String key, KV<String, Forest> kv, boolean reload) {
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
        try {

            LOG.debug("begin init dic !");
            long start = System.currentTimeMillis();
            String temp = null;
            String[] strs = null;
            Value value = null;
            try (BufferedReader br = IOUtil.getReader(PathToStream.stream(kv.getK()), "UTF-8")) {
                while ((temp = br.readLine()) != null) {
                    if (StringUtil.isNotBlank(temp)) {
                        temp = StringUtil.trim(temp);
                        strs = temp.split("\t");
                        strs[0] = strs[0].toLowerCase();
                        // 如何核心辞典存在那么就放弃
                        if (MyStaticValue.isSkipUserDefine && DATDictionary.getId(strs[0]) > 0) {
                            continue;
                        }
                        if (strs.length != 3) {
                            value = new Value(strs[0], DEFAULT_NATURE, DEFAULT_FREQ_STR);
                        } else {
                            value = new Value(strs[0], strs[1], strs[2]);
                        }
                        Library.insertWord(forest, value);
                    }
                }
            }
            LOG.info("load dic use time:" + (System.currentTimeMillis() - start) + " path is : " + kv.getK());
            kv.setV(forest);
            return forest;
        } catch (Exception e) {
            LOG.error("Init dic library error :" + e.getMessage() + ", path: " + kv.getK());
            DIC.remove(key);
            return null;
        }
    }

    /**
     * 动态添加词典
     * 
     * @param dicDefault
     * @param dicDefault2
     * @param dic2
     */
    public static void put(String key, String path, Forest forest) {
        DIC.put(key, KV.with(path, forest));
        MyStaticValue.ENV.put(key, path);
    }

    /**
     * 动态添加词典
     * 
     * @param dicDefault
     * @param dicDefault2
     * @param dic2
     */
    public static void putIfAbsent(String key, String path) {

        if (!DIC.containsKey(key)) {
            DIC.put(key, KV.with(path, (Forest) null));
        }
    }

    /**
     * 动态添加词典
     * 
     * @param dicDefault
     * @param dicDefault2
     * @param dic2
     */
    public static void put(String key, String path) {
        put(key, path, null);
    }

    /**
     * 动态添加词典
     * 
     * @param <T>
     * @param <T>
     * 
     * @param dicDefault
     * @param dicDefault2
     * @param dic2
     */
    public static synchronized Forest putIfAbsent(String key, String path, Forest forest) {

        KV<String, Forest> kv = DIC.get(key);
        if (kv != null && kv.getV() != null) {
            return kv.getV();
        }
        put(key, path, forest);
        return forest;
    }

    public static KV<String, Forest> remove(String key) {
        KV<String, Forest> kv = DIC.get(key);
        if (kv != null && kv.getV() != null) {
            kv.getV().clear();
        }
        MyStaticValue.ENV.remove(key);
        return DIC.remove(key);
    }

    public static Set<String> keys() {
        return DIC.keySet();
    }

    public static void reload(String key) {
        if (!MyStaticValue.ENV.containsKey(key)) { //如果变量中不存在直接删掉这个key不解释了
            remove(key);
        }

        putIfAbsent(key, MyStaticValue.ENV.get(key));

        KV<String, Forest> kv = DIC.get(key);

        init(key, kv, true);
    }

}
