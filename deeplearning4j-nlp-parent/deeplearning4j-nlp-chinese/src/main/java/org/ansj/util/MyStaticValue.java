package org.ansj.util;

import org.ansj.app.crf.SplitWord;
import org.ansj.dic.DicReader;
import org.ansj.dic.impl.Jdbc2Stream;
import org.ansj.domain.AnsjItem;
import org.ansj.exception.LibraryException;
import org.ansj.library.*;
import org.ansj.recognition.impl.StopRecognition;
import org.nlpcn.commons.lang.tire.domain.Forest;
import org.nlpcn.commons.lang.tire.domain.SmartForest;
import org.nlpcn.commons.lang.util.FileFinder;
import org.nlpcn.commons.lang.util.IOUtil;
import org.nlpcn.commons.lang.util.ObjConver;
import org.nlpcn.commons.lang.util.StringUtil;
import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.*;
import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.PropertyResourceBundle;
import java.util.ResourceBundle;

/**
 * 这个类储存一些公用变量.
 * 
 * @author ansj
 * 
 */
public class MyStaticValue {

    public static final Log LOG = LogFactory.getLog(MyStaticValue.class);

    // 是否开启人名识别
    public static Boolean isNameRecognition = true;

    // 是否开启数字识别
    public static Boolean isNumRecognition = true;

    // 是否数字和量词合并
    public static Boolean isQuantifierRecognition = true;

    // 是否显示真实词语
    public static Boolean isRealName = false;

    /**
     * 是否用户辞典不加载相同的词
     */
    public static boolean isSkipUserDefine = false;

    public static final Map<String, String> ENV = new HashMap<>();

    static {
        /**
         * 配置文件变量
         */
        ResourceBundle rb = null;
        try {
            rb = ResourceBundle.getBundle("ansj_library");
        } catch (Exception e) {
            try {
                File find = FileFinder.find("ansj_library.properties", 1);
                if (find != null && find.isFile()) {
                    rb = new PropertyResourceBundle(
                                    IOUtil.getReader(find.getAbsolutePath(), System.getProperty("file.encoding")));
                    LOG.info("load ansj_library not find in classPath ! i find it in " + find.getAbsolutePath()
                                    + " make sure it is your config!");
                }
            } catch (Exception e1) {
                LOG.warn("not find ansj_library.properties. reason: " + e1.getMessage());
            }
        }

        if (rb == null) {
            try {
                rb = ResourceBundle.getBundle("library");
            } catch (Exception e) {
                try {
                    File find = FileFinder.find("library.properties", 2);
                    if (find != null && find.isFile()) {
                        rb = new PropertyResourceBundle(
                                        IOUtil.getReader(find.getAbsolutePath(), System.getProperty("file.encoding")));
                        LOG.info("load library not find in classPath ! i find it in " + find.getAbsolutePath()
                                        + " make sure it is your config!");
                    }
                } catch (Exception e1) {
                    LOG.warn("not find library.properties. reason: " + e1.getMessage());
                }
            }
        }

        if (rb == null) {
            LOG.warn("not find library.properties in classpath use it by default !");
        } else {

            for (String key : rb.keySet()) {
                ENV.put(key, rb.getString(key));
                try {
                    String value = rb.getString(key);
                    if (value.startsWith("jdbc:")) { //给jdbc窜中密码做一个加密,不让密码明文在日志中
                        value = Jdbc2Stream.encryption(value);
                    }
                    LOG.info("init " + key + " to env value is : " + value);
                    Field field = MyStaticValue.class.getField(key);
                    field.set(null, ObjConver.conversion(rb.getString(key), field.getType()));
                } catch (Exception e) {
                }
            }

        }
    }

    /**
     * 人名词典
     * 
     * @return
     */
    public static BufferedReader getPersonReader() {
        return DicReader.getReader("person/person.dic");
    }

    /**
     * 机构名词典
     * 
     * @return
     */
    public static BufferedReader getCompanReader() {
        return DicReader.getReader("company/company.data");
    }

    /**
     * 机构名词典
     * 
     * @return
     */
    public static BufferedReader getNewWordReader() {
        return DicReader.getReader("newWord/new_word_freq.dic");
    }

    /**
     * 核心词典
     * 
     * @return
     */
    public static BufferedReader getArraysReader() {
        return DicReader.getReader("arrays.dic");
    }

    /**
     * 数字词典
     * 
     * @return
     */
    public static BufferedReader getNumberReader() {
        return DicReader.getReader("numberLibrary.dic");
    }

    /**
     * 英文词典
     * 
     * @return
     */
    public static BufferedReader getEnglishReader() {
        return DicReader.getReader("englishLibrary.dic");
    }

    /**
     * 词性表
     * 
     * @return
     */
    public static BufferedReader getNatureMapReader() {
        return DicReader.getReader("nature/nature.map");
    }

    /**
     * 词性关联表
     * 
     * @return
     */
    public static BufferedReader getNatureTableReader() {
        return DicReader.getReader("nature/nature.table");
    }

    /**
     * 得道姓名单字的词频词典
     * 
     * @return
     */
    public static BufferedReader getNatureClassSuffix() {
        return DicReader.getReader("nature_class_suffix.txt");
    }

    /**
     * 根据词语后缀判断词性
     * 
     * @return
     */
    public static BufferedReader getPersonFreqReader() {
        return DicReader.getReader("person/name_freq.dic");
    }

    /**
     * 名字词性对象反序列化
     * 
     * @return
     */
    @SuppressWarnings("unchecked")
    public static Map<String, int[][]> getPersonFreqMap() {
        Map<String, int[][]> map = new HashMap<>(0);
        try (InputStream inputStream = DicReader.getInputStream("person/asian_name_freq.data")) {
            ObjectInputStream objectInputStream = new ObjectInputStream(inputStream);
            map = (Map<String, int[][]>) objectInputStream.readObject();
        } catch (IOException e) {
            LOG.warn("IO异常", e);
        } catch (ClassNotFoundException e) {
            LOG.warn("找不到类", e);
        }
        return map;
    }

    /**
     * 词与词之间的关联表数据
     * 
     * @return
     */
    public static void initBigramTables() {
        try (BufferedReader reader = IOUtil.getReader(DicReader.getInputStream("bigramdict.dic"), "UTF-8")) {
            String temp = null;
            String[] strs = null;
            int freq = 0;
            while ((temp = reader.readLine()) != null) {
                if (StringUtil.isBlank(temp)) {
                    continue;
                }
                strs = temp.split("\t");
                freq = Integer.parseInt(strs[1]);
                strs = strs[0].split("@");
                AnsjItem fromItem = DATDictionary.getItem(strs[0]);

                AnsjItem toItem = DATDictionary.getItem(strs[1]);

                if (fromItem == AnsjItem.NULL && strs[0].contains("#")) {
                    fromItem = AnsjItem.BEGIN;
                }

                if (toItem == AnsjItem.NULL && strs[1].contains("#")) {
                    toItem = AnsjItem.END;
                }

                if (fromItem == AnsjItem.NULL || toItem == AnsjItem.NULL) {
                    continue;
                }

                if (fromItem.bigramEntryMap == null) {
                    fromItem.bigramEntryMap = new HashMap<Integer, Integer>();
                }

                fromItem.bigramEntryMap.put(toItem.getIndex(), freq);

            }
        } catch (NumberFormatException e) {
            LOG.warn("数字格式异常", e);
        } catch (UnsupportedEncodingException e) {
            LOG.warn("不支持的编码", e);
        } catch (IOException e) {
            LOG.warn("IO异常", e);
        }
    }

    /*
     * 外部引用为了实例化加载变量
     */
    public static Log getLog(Class<?> clazz) {
        return LogFactory.getLog(clazz);
    }

    /**
     * 增加一个词典
     * 
     * @param key
     * @param path
     * @param value
     */
    public static void putLibrary(String key, String path, Object value) {
        if (key.startsWith(DicLibrary.DEFAULT)) {
            DicLibrary.put(key, path, (Forest) value);
        } else if (key.startsWith(StopLibrary.DEFAULT)) {
            StopLibrary.put(key, path, (StopRecognition) value);
        } else if (key.startsWith(SynonymsLibrary.DEFAULT)) {
            SynonymsLibrary.put(key, path, (SmartForest) value);
        } else if (key.startsWith(AmbiguityLibrary.DEFAULT)) {
            AmbiguityLibrary.put(key, path, (Forest) value);
        } else if (key.startsWith(CrfLibrary.DEFAULT)) {
            CrfLibrary.put(key, path, (SplitWord) value);
        } else {
            throw new LibraryException(key + " type err must start with dic,stop,ambiguity,synonyms");
        }
        ENV.put(key, path);
    }

    /**
     * 懒加载一个词典
     * 
     * @param key
     * @param path
     */
    public static void putLibrary(String key, String path) {
        if (key.startsWith(DicLibrary.DEFAULT)) {
            DicLibrary.put(key, path);
        } else if (key.startsWith(StopLibrary.DEFAULT)) {
            StopLibrary.put(key, path);
        } else if (key.startsWith(SynonymsLibrary.DEFAULT)) {
            SynonymsLibrary.put(key, path);
        } else if (key.startsWith(AmbiguityLibrary.DEFAULT)) {
            AmbiguityLibrary.put(key, path);
        } else if (key.startsWith(CrfLibrary.DEFAULT)) {
            CrfLibrary.put(key, path);
        } else {
            throw new LibraryException(key + " type err must start with dic,stop,ambiguity,synonyms");
        }
        ENV.put(key, path);
    }

    /**
     * 删除一个词典
     * 
     * @param key
     */
    public static void removeLibrary(String key) {
        if (key.startsWith(DicLibrary.DEFAULT)) {
            DicLibrary.remove(key);
        } else if (key.startsWith(StopLibrary.DEFAULT)) {
            StopLibrary.remove(key);
        } else if (key.startsWith(SynonymsLibrary.DEFAULT)) {
            SynonymsLibrary.remove(key);
        } else if (key.startsWith(AmbiguityLibrary.DEFAULT)) {
            AmbiguityLibrary.remove(key);
        } else if (key.startsWith(CrfLibrary.DEFAULT)) {
            CrfLibrary.remove(key);
        } else {
            throw new LibraryException(key + " type err must start with dic,stop,ambiguity,synonyms");
        }
        ENV.remove(key);
    }

    /**
     * 重置一个词典
     * 
     * @param key
     */
    public static void reloadLibrary(String key) {
        if (key.startsWith(DicLibrary.DEFAULT)) {
            DicLibrary.reload(key);
        } else if (key.startsWith(StopLibrary.DEFAULT)) {
            StopLibrary.reload(key);
        } else if (key.startsWith(SynonymsLibrary.DEFAULT)) {
            SynonymsLibrary.reload(key);
        } else if (key.startsWith(AmbiguityLibrary.DEFAULT)) {
            AmbiguityLibrary.reload(key);
        } else if (key.startsWith(CrfLibrary.DEFAULT)) {
            CrfLibrary.reload(key);
        } else {
            throw new LibraryException(key + " type err must start with dic,stop,ambiguity,synonyms");
        }
    }
}
