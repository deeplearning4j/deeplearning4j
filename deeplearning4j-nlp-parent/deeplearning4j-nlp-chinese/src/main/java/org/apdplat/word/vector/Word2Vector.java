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

package org.apdplat.word.vector;

import org.apdplat.word.util.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.Map.Entry;
import java.util.stream.Collectors;

/**
 * 用词向量来表达一个词
 * @author 杨尚川
 */
public class Word2Vector {
    private static final Logger LOGGER = LoggerFactory.getLogger(Word2Vector.class);
    public static void main(String[] args){
        String input = "data/word.txt";
        String output = "data/vector.txt";
        String vocabulary = "data/vocabulary.txt";
        int window = 2;
        int vectorLength = 30;
        if(args.length == 3){
            input = args[0];
            output = args[1];
            vocabulary = args[2];
        }
        if(args.length == 5){
            input = args[0];
            output = args[1];
            vocabulary = args[2];
            window = Integer.parseInt(args[3]);
            vectorLength = Integer.parseInt(args[4]);
        }
        long start = System.currentTimeMillis();
        word2Vec(input, output, vocabulary, window, vectorLength);
        long cost = System.currentTimeMillis()-start;
        LOGGER.info("cost time:"+cost+" ms");        
    }

    private static void word2Vec(String input, String output, String vocabulary, int window, int vectorLength) {
        float max=(float)Runtime.getRuntime().maxMemory()/1000000;
        float total=(float)Runtime.getRuntime().totalMemory()/1000000;
        float free=(float)Runtime.getRuntime().freeMemory()/1000000;
        String pre="执行之前剩余内存:"+max+"-"+total+"+"+free+"="+(max-total+free);
        File outputFile = new File(output);
        //准备输出目录
        if(!outputFile.getParentFile().exists()){
            outputFile.getParentFile().mkdirs();
        }
        File vocabularyFile = new File(vocabulary);
        //准备输出目录
        if(!vocabularyFile.getParentFile().exists()){
            vocabularyFile.getParentFile().mkdirs();
        }
        try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(input),"utf-8"));
                BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile),"utf-8"));
                BufferedWriter vocabularyWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(vocabularyFile),"utf-8"))){
            int textLength=0;
            long start = System.currentTimeMillis();
            LOGGER.info("1、开始计算相关词");
            LOGGER.info("上下文窗口："+window);
            Map<String, List<ContextWord>> map = new HashMap<>();
            Map<String, Integer> frq = new HashMap<>();
            String line = null;
            while((line = reader.readLine()) != null){
                textLength += line.length();
                word2Vec(map, frq, line, window);
            }
            long cost = System.currentTimeMillis()-start;
            LOGGER.info("计算完毕，速度："+ textLength/cost+" 字符/毫秒，耗时："+cost/1000+" 秒，数据大小："+map.size()+"，词大小："+frq.size());
            LOGGER.info("2、开始整理数据，归一化、排序、截取TOPN");
            LOGGER.info("向量长度："+vectorLength);
            start = System.currentTimeMillis();
            normalize(map, frq, vectorLength);
            LOGGER.info("处理完毕，耗时："+ (System.currentTimeMillis()-start)/1000+" 秒，数据大小："+map.size());
            LOGGER.info("3、开始输出结果");
            start = System.currentTimeMillis();
            List<String> list = new ArrayList<>();
            for(Entry<String, List<ContextWord>> entry : map.entrySet()){
                list.add(entry.getKey()+" : "+entry.getValue().toString());
            }
            Collections.sort(list);
            for(String item : list){
                writer.write(item+"\n");
            }
            list.clear();
            List<Entry<String, Integer>> entrys = frq.entrySet().parallelStream().sorted((a,b)->b.getValue().compareTo(a.getValue())).collect(Collectors.toList());
            for(Entry<String, Integer> entry : entrys){
                vocabularyWriter.write(entry.getKey()+" "+entry.getValue()+"\n");
            }            
            LOGGER.info("输出完毕，耗时："+ (System.currentTimeMillis()-start)/1000+" 秒，数据项："+list.size());
        }catch(Exception e){
            LOGGER.error("操作错误", e);
        }
        max=(float)Runtime.getRuntime().maxMemory()/1000000;
        total=(float)Runtime.getRuntime().totalMemory()/1000000;
        free=(float)Runtime.getRuntime().freeMemory()/1000000;
        String post="执行之后剩余内存:"+max+"-"+total+"+"+free+"="+(max-total+free);
        LOGGER.info(pre);
        LOGGER.info(post);
    }
    private static void word2Vec(Map<String, List<ContextWord>> map, Map<String, Integer> frq, String line, int distance){
        String[] words = line.split(" ");
        if(words.length > 10000){
            LOGGER.info("行大小："+words.length);        
        }
        for(int i=0; i<words.length; i++){
            if(i > 0 && i % 10000 == 0){
                LOGGER.info("行处理进度: "+i/(float)words.length*100+" %");
            }
            String word = words[i];
            if(!Utils.isChineseCharAndLengthAtLeastTwo(word)){
                continue;
            }
            //计算总词频
            Integer count = frq.get(word);
            if(count == null){
                count = 1;
            }else{
                count++;
            }
            frq.put(word, count);
            //计算word的上下文关联词
            for(int j=1; j<=distance; j++){                
                //计算word的上文
                int index = i-j;
                contextWord(words, index, j, word, map);
                //计算word的下文
                index = i+j;
                contextWord(words, index, j, word, map);
            }
        }
    }
    /**
     * 计算词的相关词
     * @param words 词数组
     * @param index 相关词索引
     * @param distance 词距
     * @param word 核心词
     * @param map 
     */
    private static void contextWord(String[] words, int index, int distance, String word, Map<String, List<ContextWord>> map){
        String _word = null;
        if(index > -1 && index < words.length){
            _word = words[index];                    
        }
        if(_word != null && Utils.isChineseCharAndLengthAtLeastTwo(_word)){
            addToMap(map, word, _word, distance);
        }
    }
    private static void addToMap(Map<String, List<ContextWord>> map, String word, String _word, int distance){
        List<ContextWord> value = map.get(word);
        if(value == null){
            value = new ArrayList<>();
            map.put(word, value);
        }
        float s = (float)1/distance;
        boolean find=false;
        for(ContextWord item : value){
            if(item.getWord().equals(_word)){
                float score = item.getScore()+s;
                item.setScore(score);
                find = true;
                break;
            }
        }
        if(!find){
            ContextWord item = new ContextWord(_word, s);
            value.add(item);
        }
    }
    private static void normalize(Map<String, List<ContextWord>> map, Map<String, Integer> frq, int count) {        
        for(String key : map.keySet()){
            List<ContextWord> value = map.get(key);
            //分值归一化
            float max=0;
            for(ContextWord word : value){
                //加入词频信息，重新计算分值
                //float score = word.getScore() / (float)Math.sqrt(frq.get(word.getWord()));
                //word.setScore(score);
                if(word.getScore() > max){
                    max = word.getScore();
                }
            }
            for(ContextWord word : value){
                word.setScore(word.getScore()/max);
            }
            //排序
            Collections.sort(value);
            int len = value.size();
            //截取需要的TOPN
            if(len > count){
                value = value.subList(0, count);
            }
            map.put(key, value);
        }
    }
    private static class ContextWord implements Comparable{
        private String word;
        private float score;
        public ContextWord(String word, Float score) {
            this.word = word;
            this.score = score;
        }
        @Override
        public String toString() {
            return word + " " + score;
        }        
        public String getWord() {
            return word;
        }
        public void setWord(String word) {
            this.word = word;
        }
        public float getScore() {
            return score;
        }
        public void setScore(float score) {
            this.score = score;
        }        
        @Override
        public int compareTo(Object o) {
            float target = ((ContextWord)o).getScore();
            if(this.getScore() < target){
                return 1;
            }
            if(this.getScore() == target){
                return 0;
            }
            return -1;
        }
    }
}