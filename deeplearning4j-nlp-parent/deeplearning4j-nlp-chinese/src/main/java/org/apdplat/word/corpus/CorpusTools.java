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

package org.apdplat.word.corpus;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import org.apdplat.word.dictionary.DictionaryTools;
import org.apdplat.word.util.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 语料库工具
 * 用于构建二元模型和三元模型并做进一步的分析处理
 * 同时把语料库中的新词加入词典
 * @author 杨尚川
 */
public class CorpusTools {
    private static final Logger LOGGER = LoggerFactory.getLogger(CorpusTools.class);
    private static final Map<String, Integer> BIGRAM = new TreeMap<>();
    private static final Map<String, Integer> TRIGRAM = new TreeMap<>();
    private static final AtomicInteger WORD_COUNT = new AtomicInteger();   
    private static final AtomicInteger CHAR_COUNT = new AtomicInteger();    
    private static final AtomicInteger LINES_COUNT = new AtomicInteger();    
    private static final Map<String, Integer> WORDS = new TreeMap<>();
    private static final Set<String> PHRASES = new HashSet<>();
    
    public static void main(String[] args){
        process();
    }
    /**
     * 分析语料库
     * 分析处理并保存二元模型
     * 分析处理并保存三元模型
     * 分析处理并保存语料库中提取出来的词
     * 将新提取的词和原来的词典合并
     * 二元和三元模型规范化
     */
    private static void process(){
        //分析语料库
        analyzeCorpus();
        //分析处理并保存短语结构
        processPhrase();
        //分析处理并保存二元模型
        processBiGram();
        BIGRAM.clear();
        //分析处理并保存三元模型
        processTriGram();
        TRIGRAM.clear();
        //分析处理并保存语料库中提取出来的词
        processWords();
        WORDS.clear();
        //将新提取的词和原来的词典合并
        mergeWordsWithOldDic();
        //移除词典中的短语结构
        DictionaryTools.removePhraseFromDic("target/phrase.txt", "src/main/resources/dic.txt");
    }    
    /**
     * 分析语料库
     */
    private static void analyzeCorpus(){
        String zipFile = "src/main/resources/corpus/corpora.zip";        
        LOGGER.info("开始分析语料库");
        long start = System.currentTimeMillis();        
        try{
            analyzeCorpus(zipFile);
        } catch (IOException ex) {
            LOGGER.info("分析语料库失败：", ex);
        }
        long cost = System.currentTimeMillis() - start;
        LOGGER.info("完成分析语料库，耗时："+cost+"毫秒");
        LOGGER.info("语料库行数为："+LINES_COUNT.get()+"，总字符数目为："+CHAR_COUNT.get()+"，总词数目为："+WORD_COUNT.get()+"，不重复词数目为："+WORDS.size());
    }
    /**
     * 分析语料库
     * @param zipFile 压缩的语料库
     * @throws IOException 
     */
    private static void analyzeCorpus(String zipFile) throws IOException{
        try (FileSystem fs = FileSystems.newFileSystem(Paths.get(zipFile), CorpusTools.class.getClassLoader())) {
            for(Path path : fs.getRootDirectories()){                
                LOGGER.info("处理目录："+path);
                Files.walkFileTree(path, new SimpleFileVisitor<Path>(){

                    @Override
                    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                        LOGGER.info("处理文件："+file);
                        // 拷贝到本地文件系统
                        Path temp = Paths.get("target/corpus-"+System.currentTimeMillis()+".txt");
                        Files.copy(file, temp, StandardCopyOption.REPLACE_EXISTING);
                        constructNGram(temp);
                        return FileVisitResult.CONTINUE;
                    }
                    
                });
            }
        }
    }
    /**
     * 构建二元模型、三元模型、短语结构，统计字符数、词数
     * 构建不重复词列表
     * @param file 
     */
    private static void constructNGram(Path file){
        try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file.toFile()),"utf-8"));){
            String line;
            while( (line = reader.readLine()) != null ){
                LINES_COUNT.incrementAndGet();
                //去除首尾空白字符
                line = line.trim();
                //忽略空行
                if(!"".equals(line)){
                    //词和词之间以空格隔开
                    String[] words = line.split("\\s+");
                    if(words == null){
                        //忽略不符合规范的行
                        continue;
                    }
                    List<String> list = new ArrayList<>();
                    StringBuilder phrase = new StringBuilder();
                    int phraseCount=0;
                    boolean find=false;
                    for(String word : words){
                        String[] attr = word.split("/");
                        if(attr == null || attr.length < 1){
                            //忽略不符合规范的词
                            continue;
                        }
                        if(attr[0].trim().startsWith("[")){
                            find = true;
                        }                        
                        //去掉[和]
                        String item = attr[0].replace("[", "").replace("]", "").trim();
                        list.add(item);
                        //不重复词
                        addWord(item);
                        //词数目
                        WORD_COUNT.incrementAndGet();
                        //字符数目
                        CHAR_COUNT.addAndGet(item.length());
                        if(find){
                            phrase.append(item).append(" ");
                            phraseCount++;
                        }
                        //短语标注错误
                        if(phraseCount > 10){
                            find = false;
                            phraseCount = 0;
                            phrase.setLength(0);
                        }
                        if(find && attr.length > 1 && attr[1].trim().endsWith("]")){
                            find = false;
                            PHRASES.add(phrase.toString());
                            phrase.setLength(0);
                        }
                    }
                    //计算bigram模型
                    int len = list.size();
                    if(len < 2){
                        continue;
                    }
                    for(int i=0; i<len-1; i++){
                        String first = list.get(i);
                        if(!Utils.isChineseCharAndLengthAtLeastOne(first)){
                            continue;
                        }
                        String second = list.get(i+1);
                        if(!Utils.isChineseCharAndLengthAtLeastOne(second)){
                            //跳过一个词
                            i++;
                            continue;
                        }
                        //忽略长度均为1的模型
                        if(first.length() < 2 && second.length() < 2){
                            continue;
                        }
                        String key = first+":"+second;
                        Integer value = BIGRAM.get(key);
                        if(value == null){
                            value = 1;
                        }else{
                            value++;
                        }
                        BIGRAM.put(key, value);
                    }
                    if(len < 3){
                        continue;
                    }
                    //计算trigram模型
                    for(int i=0; i<len-2; i++){
                        String first = list.get(i);
                        if(!Utils.isChineseCharAndLengthAtLeastOne(first)){
                            continue;
                        }
                        String second = list.get(i+1);
                        if(!Utils.isChineseCharAndLengthAtLeastOne(second)){
                            //跳过一个词
                            i++;
                            continue;
                        }
                        String third = list.get(i+2);
                        if(!Utils.isChineseCharAndLengthAtLeastOne(third)){
                            //跳过二个词
                            i += 2;
                            continue;
                        }
                        //忽略长度均为1的模型
                        if(first.length() < 2 && second.length() < 2 && third.length() < 2){
                            continue;
                        }
                        String key = first+":"+second+":"+third;
                        Integer value = TRIGRAM.get(key);
                        if(value == null){
                            value = 1;
                        }else{
                            value++;
                        }
                        TRIGRAM.put(key, value);
                    }                    
                }
            }
        }catch(Exception e){
            LOGGER.info("分析语料库 "+file+" 失败：", e);
        }        
    }
    private static void addWord(String word){
        Integer value = WORDS.get(word);
        if(value == null){
            value = 1;
        }else{
            value++;
        }
        WORDS.put(word, value);
    }
    /**
     * 分析处理并存储二元模型
     * 移除二元模型出现频率为1的情况
     * 排序后保存到bigram.txt文件
     */
    private static void processBiGram() {
        //移除二元模型出现频率为1的情况
        Iterator<String> iter = BIGRAM.keySet().iterator();
        while(iter.hasNext()){
            String key = iter.next();
            if(BIGRAM.get(key) < 2){
                iter.remove();
            }
        }
        try(BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("src/main/resources/bigram.txt"),"utf-8"))){
            for(Entry<String, Integer> item : BIGRAM.entrySet()){
                writer.write(item.getKey()+" "+item.getValue()+"\n");
            }
        }catch(Exception e){
            LOGGER.info("保存bigram模型失败：", e);
        }
    }
    /**
     * 分析处理并存储三元模型
     * 移除三元模型出现频率为1的情况
     * 排序后保存到trigram.txt文件
     */
    private static void processTriGram() {
        //移除三元模型出现频率为1的情况
        Iterator<String> iter = TRIGRAM.keySet().iterator();
        while(iter.hasNext()){
            String key = iter.next();
            if(TRIGRAM.get(key) < 2){
                iter.remove();
            }
        }
        try(BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("src/main/resources/trigram.txt"),"utf-8"))){
            for(Entry<String, Integer> item : TRIGRAM.entrySet()){
                writer.write(item.getKey()+" "+item.getValue()+"\n");
            }
        }catch(Exception e){
            LOGGER.info("保存trigram模型失败：", e);
        }
    }
    /**
     * 分析处理并存储不重复词
     */
    private static void processWords() {
        try(BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("target/dic.txt"),"utf-8"));
                BufferedWriter writerFreq = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("target/dic_with_freq.txt"),"utf-8"))){
            for(String word : WORDS.keySet()){
                writer.write(word+"\n");
            }
            List<Entry<String, Integer>> entrys = WORDS.entrySet().parallelStream().sorted((a,b)->b.getValue().compareTo(a.getValue())).collect(Collectors.toList());
            for(Entry<String, Integer> entry : entrys){
                writerFreq.write(entry.getKey()+" "+ entry.getValue()+"\n");
            }
        }catch(Exception e){
            LOGGER.info("保存词典文件失败：", e);
        }
    }
    /**
     * 和旧的词典进行合并
     */
    private static void mergeWordsWithOldDic() {
        List<String> sources = new ArrayList<>();
        sources.add("target/dic.txt");
        sources.add("src/main/resources/dic.txt");
        String target = "src/main/resources/dic.txt";
        try {
            DictionaryTools.merge(sources, target);
        } catch (IOException ex) {
            LOGGER.info("和现有词典合并失败：", ex);
        }
    }
    /**
     * 生成短语结构
     */
    private static void processPhrase() {
        try(BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("target/phrase.txt"),"utf-8"))){
            List<String> list = new ArrayList<>();
            list.addAll(PHRASES);
            PHRASES.clear();
            Collections.sort(list);
            for(String phrase : list){
                String p = phrase.replaceAll("\\s+", "");
                writer.write(p+"="+phrase.trim()+"\n");
            }
            list.clear();
        }catch(Exception e){
            LOGGER.info("保存短语结构失败：", e);
        }
    }
}
