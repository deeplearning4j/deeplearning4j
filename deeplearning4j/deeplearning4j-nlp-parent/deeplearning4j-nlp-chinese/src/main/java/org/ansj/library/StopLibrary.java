package org.ansj.library;

import org.ansj.dic.PathToStream;
import org.ansj.domain.KV;
import org.ansj.recognition.impl.StopRecognition;
import org.ansj.util.MyStaticValue;
import org.nlpcn.commons.lang.util.IOUtil;
import org.nlpcn.commons.lang.util.StringUtil;
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.BufferedReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class StopLibrary {

    private static final Log LOG = LogFactory.getLog();

    public static final String DEFAULT = "stop";

    // 用户自定义词典
    private static final Map<String, KV<String, StopRecognition>> STOP = new HashMap<>();

    static {
        for (Entry<String, String> entry : MyStaticValue.ENV.entrySet()) {
            if (entry.getKey().startsWith(DEFAULT)) {
                put(entry.getKey(), entry.getValue());
            }
        }
        putIfAbsent(DEFAULT, "library/stop.dic");
    }

    /**
     * 词性过滤
     * 
     * @param key
     * @param stopNatures
     */
    public static void insertStopNatures(String key, String... filterNatures) {
        StopRecognition fr = get(key);
        fr.insertStopNatures(filterNatures);
    }

    /**
     * 正则过滤
     * 
     * @param key
     * @param regexes
     */
    public static void insertStopRegexes(String key, String... regexes) {
        StopRecognition fr = get(key);
        fr.insertStopRegexes(regexes);
    }

    /**
     * 增加停用词
     * 
     * @param key
     * @param regexes
     */
    public static void insertStopWords(String key, String... stopWords) {
        StopRecognition fr = get(key);
        fr.insertStopWords(stopWords);
    }

    /**
     * 增加停用词
     * 
     * @param key
     * @param regexes
     */
    public static void insertStopWords(String key, List<String> stopWords) {
        StopRecognition fr = get(key);
        fr.insertStopWords(stopWords);
    }

    public static StopRecognition get() {
        return get(DEFAULT);
    }

    /**
     * 根据模型名称获取crf模型
     * 
     * @param modelName
     * @return
     */
    public static StopRecognition get(String key) {
        KV<String, StopRecognition> kv = STOP.get(key);

        if (kv == null) {
            if (MyStaticValue.ENV.containsKey(key)) {
                putIfAbsent(key, MyStaticValue.ENV.get(key));
                return get(key);
            }
            LOG.warn("STOP " + key + " not found in config ");
            return null;
        }
        StopRecognition stopRecognition = kv.getV();
        if (stopRecognition == null) {
            stopRecognition = init(key, kv, false);
        }
        return stopRecognition;

    }

    /**
     * 用户自定义词典加载
     * 
     * @param key
     * @param path
     * @return
     */
    private synchronized static StopRecognition init(String key, KV<String, StopRecognition> kv, boolean reload) {
        StopRecognition stopRecognition = kv.getV();

        if (stopRecognition != null) {
            if (reload) {
                stopRecognition.clear();
            } else {
                return stopRecognition;
            }
        } else {
            stopRecognition = new StopRecognition();
        }

        try {
            LOG.debug("begin init FILTER !");
            long start = System.currentTimeMillis();
            String temp = null;
            String[] strs = null;
            try (BufferedReader br = IOUtil.getReader(PathToStream.stream(kv.getK()), "UTF-8")) {
                while ((temp = br.readLine()) != null) {
                    if (StringUtil.isNotBlank(temp)) {
                        temp = StringUtil.trim(temp);
                        strs = temp.split("\t");

                        if (strs.length == 1) {
                            stopRecognition.insertStopWords(strs[0]);
                        } else {
                            switch (strs[1]) {
                                case "nature":
                                    stopRecognition.insertStopNatures(strs[0]);
                                    break;
                                case "regex":
                                    stopRecognition.insertStopRegexes(strs[0]);
                                    break;
                                default:
                                    stopRecognition.insertStopWords(strs[0]);
                                    break;
                            }
                        }

                    }
                }
            }
            LOG.info("load stop use time:" + (System.currentTimeMillis() - start) + " path is : " + kv.getK());
            kv.setV(stopRecognition);
            return stopRecognition;
        } catch (Exception e) {
            LOG.error("Init Stop library error :" + e.getMessage() + ", path: " + kv.getK());
            STOP.remove(key);
            return null;
        }
    }

    /**
     * 动态添加词典
     * 
     * @param FILTERDefault
     * @param FILTERDefault2
     * @param FILTER2
     */
    public static void put(String key, String path, StopRecognition stopRecognition) {
        STOP.put(key, KV.with(path, stopRecognition));
        MyStaticValue.ENV.put(key, path);
    }

    /**
     * 动态添加词典
     * 
     * @param FILTERDefault
     * @param FILTERDefault2
     * @param FILTER2
     */
    public static void putIfAbsent(String key, String path) {
        if (!STOP.containsKey(key)) {
            STOP.put(key, KV.with(path, (StopRecognition) null));
        }
    }

    /**
     * 动态添加词典
     * 
     * @param FILTERDefault
     * @param FILTERDefault2
     * @param FILTER2
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
     * @param FILTERDefault
     * @param FILTERDefault2
     * @param FILTER2
     */
    public static synchronized StopRecognition putIfAbsent(String key, String path, StopRecognition stopRecognition) {
        KV<String, StopRecognition> kv = STOP.get(key);
        if (kv != null && kv.getV() != null) {
            return kv.getV();
        }
        put(key, path, stopRecognition);
        return stopRecognition;
    }

    public static KV<String, StopRecognition> remove(String key) {
        KV<String, StopRecognition> kv = STOP.get(key);
        if (kv != null && kv.getV() != null) {
            kv.getV().clear();
        }
        MyStaticValue.ENV.remove(key);
        return STOP.remove(key);
    }

    public static Set<String> keys() {
        return STOP.keySet();
    }

    public static void reload(String key) {

        if (!MyStaticValue.ENV.containsKey(key)) { //如果变量中不存在直接删掉这个key不解释了
            remove(key);
        }

        putIfAbsent(key, MyStaticValue.ENV.get(key));

        KV<String, StopRecognition> kv = STOP.get(key);

        init(key, kv, true);
    }

}
