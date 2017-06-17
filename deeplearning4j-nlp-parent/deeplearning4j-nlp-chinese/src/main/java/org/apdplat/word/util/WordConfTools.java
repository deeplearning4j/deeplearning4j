/**
 * 
 * APDPlat - Application Product Development Platform
 * Copyright (c) 2013, 杨尚川, yang-shangchuan@qq.com
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

package org.apdplat.word.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 获取配置信息的工具类
 * @author 杨尚川
 */
public class WordConfTools {
    private static final Logger LOGGER = LoggerFactory.getLogger(WordConfTools.class);
    private static final Map<String, String> conf = new HashMap<>();
    public static void set(String key, String value){
        conf.put(key, value);
    }
    public static boolean getBoolean(String key, boolean defaultValue){
        String value = conf.get(key) == null ? Boolean.valueOf(defaultValue).toString() : conf.get(key);
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("获取配置项：" + key + "=" + value);
        }
        return value.contains("true");
    }
    public static boolean getBoolean(String key){
        return getBoolean(key, false);
    }
    public static int getInt(String key, int defaultValue){
        int value = conf.get(key) == null ? defaultValue : Integer.parseInt(conf.get(key).trim());
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("获取配置项：" + key + "=" + value);
        }
        return value;
    }
    public static int getInt(String key){
        return getInt(key, -1);
    }
    public static String get(String key, String defaultValue){
        String value = conf.get(key) == null ? defaultValue : conf.get(key);
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("获取配置项：" + key + "=" + value);
        }
        return value;
    }
    public static String get(String key){
        String value = conf.get(key);
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("获取配置项：" + key + "=" + value);
        }
        return value;
    }
    static{
        reload();
    }
    /**
     * 重新加载配置文件
     */
    public static void reload(){
        conf.clear();
        LOGGER.info("开始加载配置文件");
        long start = System.currentTimeMillis();
        loadConf("word.conf");
        loadConf("word.local.conf");
        checkSystemProperties();
        long cost = System.currentTimeMillis() - start;
        LOGGER.info("配置文件加载完毕，耗时"+cost+" 毫秒，配置项数目："+conf.size());
        LOGGER.info("配置信息：");
        AtomicInteger i = new AtomicInteger();
        conf.keySet().stream().sorted().forEach(key -> {
            LOGGER.info(i.incrementAndGet()+"、"+key+"="+conf.get(key));
        });
    }
    /**
     * 强制覆盖默认配置
     * @param confFile 配置文件路径
     */
    public static void forceOverride(String confFile) {
        File file = new File(confFile);
        try(InputStream in = new FileInputStream(file)){
            LOGGER.info("使用配置文件 "+file.getAbsolutePath()+" 强制覆盖默认配置");
            loadConf(in);
        } catch (Exception ex) {
            LOGGER.error("强制覆盖默认配置失败：", ex);
        }
        int i=1;
        for(String key : conf.keySet()){
            LOGGER.info((i++)+"、"+key+"="+conf.get(key));
        }
    }
    /**
     * 加载配置文件
     * @param confFile 类路径下的配置文件 
     */
    private static void loadConf(String confFile) {
        InputStream in = WordConfTools.class.getClassLoader().getResourceAsStream(confFile);
        if(in == null){
            LOGGER.info("未找到配置文件："+confFile);
            return;
        }
        LOGGER.info("加载配置文件："+confFile);
        loadConf(in);
    }
    /**
     * 加载配置文件
     * @param in 文件输入流
     */
    private static void loadConf(InputStream in) {
        try(BufferedReader reader = new BufferedReader(new InputStreamReader(in, "utf-8"))){
            String line;
            while((line = reader.readLine()) != null){
                line = line.trim();                
                if("".equals(line) || line.startsWith("#")){
                    continue;
                }
                int index = line.indexOf("=");
                if(index==-1){
                    LOGGER.error("错误的配置："+line);
                    continue;
                }
                //有K V
                if(index>0 && line.length()>index+1) {
                    String key = line.substring(0, index).trim();
                    String value = line.substring(index + 1, line.length()).trim();
                    conf.put(key, value);
                }
                //有K无V
                else if(index>0 && line.length()==index+1) {
                    String key = line.substring(0, index).trim();
                    conf.put(key, "");
                }else{
                    LOGGER.error("错误的配置："+line);
                }
            }
        } catch (IOException ex) {
            System.err.println("配置文件加载失败:"+ex.getMessage());
            throw new RuntimeException(ex);
        }
    }
    /**
     * 使用系统属性覆盖配置文件
     */
    private static void checkSystemProperties() {
        for(String key : conf.keySet()){
            String value = System.getProperty(key);
            if(value != null){
                conf.put(key, value);
                LOGGER.info("系统属性覆盖默认配置："+key+"="+value);
            }
        }
    }
    public static void main(String[] args){
    }
}