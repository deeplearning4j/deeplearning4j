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

package org.apdplat.word.recognition;

import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.apdplat.word.segmentation.Word;
import org.apdplat.word.util.AutoDetector;
import org.apdplat.word.util.ResourceLoader;
import org.apdplat.word.util.WordConfTools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 停用词判定
 * 通过系统属性及配置文件指定停用词词典（stopwords.path）
 * 指定方式一，编程指定（高优先级）：
 *      WordConfTools.set("stopwords.path", "classpath:stopwords.txt");
 * 指定方式二，Java虚拟机启动参数（中优先级）：
 *      java -Dstopwords.path=classpath:stopwords.txt
 * 指定方式三，配置文件指定（低优先级）：
 *      在类路径下的word.conf中指定配置信息
 *      stopwords.path=classpath:stopwords.txt
 * 如未指定，则默认使用停用词词典文件（类路径下的stopwords.txt）
 * @author 杨尚川
 */
public class StopWord {
    private static final Logger LOGGER = LoggerFactory.getLogger(StopWord.class);
    private static final Set<String> stopwords = new HashSet<>();
    static{
        reload();
    }
    public static void reload(){
        AutoDetector.loadAndWatch(new ResourceLoader(){

            @Override
            public void clear() {
                stopwords.clear();
            }

            @Override
            public void load(List<String> lines) {
                LOGGER.info("初始化停用词");
                for(String line : lines){
                    if(!isStopChar(line)){
                        stopwords.add(line);
                    }
                }
                LOGGER.info("停用词初始化完毕，停用词个数："+stopwords.size());
            }

            @Override
            public void add(String line) {
                if(!isStopChar(line)){
                    stopwords.add(line);
                }
            }

            @Override
            public void remove(String line) {
                if(!isStopChar(line)){
                    stopwords.remove(line);
                }
            }
        
        }, WordConfTools.get("stopwords.path", "classpath:stopwords.txt"));
    }
    /**
     * 如果词的长度为一且不是中文字符和数字，则认定为停用词
     * @param word
     * @return 
     */
    private static boolean isStopChar(String word){
        if(word.length() == 1){
            char _char = word.charAt(0);
            if(_char < 48){
                return true;
            }
            if(_char > 57 && _char < 19968){
                return true;
            }
            if(_char > 40869){
                return true;
            }
        }
        return false;
    }
    /**
     * 判断一个词是否是停用词
     * @param word
     * @return 
     */
    public static boolean is(String word){      
        if(word == null){
            return false;
        }
        word = word.trim();
        return isStopChar(word) || stopwords.contains(word);
    }
    /**
     * 停用词过滤，删除输入列表中的停用词
     * @param words 词列表
     */
    public static void filterStopWords(List<Word> words){
        Iterator<Word> iter = words.iterator();
        while(iter.hasNext()){
            Word word = iter.next();
            if(is(word.getText())){
                //去除停用词
                if(LOGGER.isDebugEnabled()) {
                    LOGGER.debug("去除停用词：" + word.getText());
                }
                iter.remove();
            }
        }
    }

    public static void main(String[] args){
        LOGGER.info("停用词：");
        int i=1;
        for(String w : stopwords){
            LOGGER.info((i++)+" : "+w);
        }
    }
}