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

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import org.apdplat.word.segmentation.PartOfSpeech;
import org.apdplat.word.segmentation.Word;
import org.apdplat.word.tagging.PartOfSpeechTagging;
import org.apdplat.word.util.AutoDetector;
import org.apdplat.word.util.ResourceLoader;
import org.apdplat.word.util.WordConfTools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 人名识别
 * @author 杨尚川
 */
public class PersonName {
    private static final Logger LOGGER = LoggerFactory.getLogger(PersonName.class);
    private static final Set<String> SURNAME_1 =new HashSet<>();
    private static final Set<String> SURNAME_2 =new HashSet<>();
    private static final Map<String, Integer> POS_SEQ=new HashMap<>();
    static{
        reload();
    }
    public static void reload(){
        AutoDetector.loadAndWatch(new ResourceLoader() {

            @Override
            public void clear() {
                SURNAME_1.clear();
                SURNAME_2.clear();
                POS_SEQ.clear();
            }

            @Override
            public void load(List<String> lines) {
                LOGGER.info("初始化百家姓");
                for (String line : lines) {
                    if (line.length() == 1) {
                        SURNAME_1.add(line);
                    } else if (line.length() == 2) {
                        SURNAME_2.add(line);
                    } else if (line.startsWith("pos_seq=")) {
                        String[] attr = line.split("=");
                        POS_SEQ.put(attr[1].trim().replaceAll("\\s", " "), Integer.parseInt(attr[2]));
                    } else {
                        LOGGER.error("错误的姓：" + line);
                    }
                }
                LOGGER.info("百家姓初始化完毕，单姓个数：" + SURNAME_1.size() + "，复姓个数：" + SURNAME_2.size());
            }

            @Override
            public void add(String line) {
                if (line.length() == 1) {
                    SURNAME_1.add(line);
                } else if (line.length() == 2) {
                    SURNAME_2.add(line);
                } else if (line.startsWith("pos_seq=")) {
                    String[] attr = line.split("=");
                    POS_SEQ.put(attr[1].trim().replaceAll("\\s", " "), Integer.parseInt(attr[2]));
                } else {
                    LOGGER.error("错误的姓：" + line);
                }
            }

            @Override
            public void remove(String line) {
                if (line.length() == 1) {
                    SURNAME_1.remove(line);
                } else if (line.length() == 2) {
                    SURNAME_2.remove(line);
                } else if (line.startsWith("pos_seq=")) {
                    String[] attr = line.split("=");
                    POS_SEQ.remove(attr[1].trim().replaceAll("\\s", " "));
                } else {
                    LOGGER.error("错误的姓：" + line);
                }
            }

        }, WordConfTools.get("surname.path", "classpath:surname.txt"));
    }
    /**
     * 获取所有的姓
     * @return 有序列表
     */
    public static List<String> getSurnames(){
        List<String> result = new ArrayList<>();
        result.addAll(SURNAME_1);
        result.addAll(SURNAME_2);
        Collections.sort(result);
        return result;
    }
    /**
     * 如果文本为人名，则返回姓
     * @param text 文本
     * @return 姓或空文本
     */
    public static String getSurname(String text){
        if(is(text)){
            //优先识别复姓
            if(isSurname(text.substring(0, 2))){
                return text.substring(0, 2);
            }
            if(isSurname(text.substring(0, 1))){
                return text.substring(0, 1);
            }
        }
        return "";
    }
    /**
     * 判断文本是不是百家姓
     * @param text 文本
     * @return 是否
     */
    public static boolean isSurname(String text){
        return SURNAME_1.contains(text) || SURNAME_2.contains(text);
    }
    /**
     * 人名判定
     * @param text 文本
     * @return 是或否
     */
    public static boolean is(String text){
        int len = text.length();
        //单姓为二字或三字
        //复姓为三字或四字
        if(len < 2){
            //长度小于2肯定不是姓名
            return false;
        }
        if(len == 2){
            //如果长度为2，则第一个字符必须是姓
            return SURNAME_1.contains(text.substring(0, 1));
        }
        if(len == 3){
            //如果长度为3
            //要么是单姓
            //要么是复姓
            return SURNAME_1.contains(text.substring(0, 1)) || SURNAME_2.contains(text.substring(0, 2));
        }
        if(len == 4){
            //如果长度为4，只能是复姓
            return SURNAME_2.contains(text.substring(0, 2));
        }
        return false;
    }
    /**
     * 对分词结果进行处理，识别人名
     * @param words 待识别分词结果
     * @return 识别后的分词结果
     */
    public static List<Word> recognize(List<Word> words){
        int len = words.size();
        if(len < 2){
            return words;
        }
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("人名识别：" + words);
        }
        List<List<Word>> select = new ArrayList<>();
        List<Word> result = new ArrayList<>();
        for(int i=0; i<len-1; i++){
            String word = words.get(i).getText();
            if(isSurname(word)){
                result.addAll(recognizePersonName(words.subList(i, words.size())));
                select.add(result);
                result = new ArrayList<>(words.subList(0, i+1));
            }else{
                result.add(new Word(word));
            }
        }
        if(select.isEmpty()){
            return words;
        }
        if(select.size()==1){
            return select.get(0);
        }
        return selectBest(select);
    }

    /**
     * 使用词性序列从多个人名中选择一个最佳的
     * @param candidateWords
     * @return
     */
    private static List<Word> selectBest(List<List<Word>> candidateWords){
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("开始从多个识别结果中选择一个最佳的结果:{}", candidateWords);
        }
        Map<List<Word>, Integer> map = new ConcurrentHashMap<>();
        AtomicInteger i = new AtomicInteger();
        candidateWords.stream().forEach(candidateWord -> {
            if(LOGGER.isDebugEnabled()) {
                LOGGER.debug(i.incrementAndGet() + "、开始处理：" + candidateWord);
            }
            //词性标注
            PartOfSpeechTagging.process(candidateWord);
            //根据词性标注的结果进行评分
            StringBuilder seq = new StringBuilder();
            candidateWord.forEach(word -> seq.append(word.getPartOfSpeech().getPos().charAt(0)).append(" "));
            String seqStr = seq.toString();
            AtomicInteger score = new AtomicInteger();
            if(LOGGER.isDebugEnabled()) {
                LOGGER.debug("词序列：{} 的词性序列：{}", candidateWord, seqStr);
            }
            POS_SEQ.keySet().parallelStream().forEach(pos_seq -> {
                if (seqStr.contains(pos_seq)) {
                    int sc = POS_SEQ.get(pos_seq);
                    if(LOGGER.isDebugEnabled()) {
                        LOGGER.debug(pos_seq + "词序增加分值：" + sc);
                    }
                    score.addAndGet(sc);
                }
            });
            score.addAndGet(-candidateWord.size());
            if(LOGGER.isDebugEnabled()) {
                LOGGER.debug("长度的负值也作为分值：" + (-candidateWord.size()));
                LOGGER.debug("评分结果：" + score.get());
            }
            map.put(candidateWord, score.get());
        });
        //选择分值最高的
        List<Word> result = map.entrySet().parallelStream().sorted((a,b)->b.getValue().compareTo(a.getValue())).map(e->e.getKey()).collect(Collectors.toList()).get(0);
        if(LOGGER.isDebugEnabled()){
            LOGGER.debug("选择结果："+result);
        }
        return result;
    }
    private static List<Word> recognizePersonName(List<Word> words){
        int len = words.size();
        if(len < 2){
            return words;
        }
        List<Word> result = new ArrayList<>();
        for(int i=0; i<len-1; i++){
            String second = words.get(i+1).getText();
            if(second.length() > 1){
                result.add(new Word(words.get(i).getText()));
                result.add(new Word(words.get(i+1).getText()));
                i++;
                if(i == len-2){
                    result.add(new Word(words.get(i+1).getText()));
                }
                continue;
            }
            String first = words.get(i).getText();
            if(isSurname(first)){
                String third = "";
                if(i+2 < len && words.get(i+2).getText().length()==1){
                    third = words.get(i+2).getText();                    
                }
                String text = first+second+third;
                if(is(text)){
                    if(LOGGER.isDebugEnabled()) {
                        LOGGER.debug("识别到人名：" + text);
                    }
                    Word word = new Word(text);
                    //词性定义参见配置文件word.conf中的定义part.of.speech.des.path=classpath:part_of_speech_des.txt
                    word.setPartOfSpeech(PartOfSpeech.valueOf("nr"));
                    result.add(word);
                    i++;
                    if(!"".equals(third)){
                        i++;
                    }
                }else{
                    result.add(new Word(first));
                }
            }else{
                result.add(new Word(first));
            }
            if(i == len-2){
                result.add(new Word(words.get(i+1).getText()));
            }
        }
        return result;
    }
    public static void main(String[] args){
        int i=1;
        for(String str : SURNAME_1){
            LOGGER.info((i++)+" : "+str);
        }
        for(String str : SURNAME_2){
            LOGGER.info((i++)+" : "+str);
        }
        LOGGER.info("杨尚川："+is("杨尚川"));
        LOGGER.info("欧阳飞燕："+is("欧阳飞燕"));
        LOGGER.info("令狐冲："+is("令狐冲"));
        LOGGER.info("杨尚川爱读书："+is("杨尚川爱读书"));
        List<Word> test = new ArrayList<>();
        test.add(new Word("快"));
        test.add(new Word("来"));
        test.add(new Word("看"));
        test.add(new Word("杨"));
        test.add(new Word("尚"));
        test.add(new Word("川"));
        test.add(new Word("表演"));
        test.add(new Word("魔术"));
        test.add(new Word("了"));
        LOGGER.info(recognize(test).toString());
        
        test = new ArrayList<>();
        test.add(new Word("李"));
        test.add(new Word("世"));
        test.add(new Word("明"));
        test.add(new Word("的"));
        test.add(new Word("昭仪"));
        test.add(new Word("欧阳"));
        test.add(new Word("飞"));
        test.add(new Word("燕"));
        test.add(new Word("其实"));
        test.add(new Word("很"));
        test.add(new Word("厉害"));
        test.add(new Word("呀"));
        test.add(new Word("！"));
        test.add(new Word("比"));
        test.add(new Word("公孙"));
        test.add(new Word("黄"));
        test.add(new Word("后"));
        test.add(new Word("牛"));
        LOGGER.info(recognize(test).toString());
                
        test = new ArrayList<>();
        test.add(new Word("发展"));
        test.add(new Word("中国"));
        test.add(new Word("家兔"));
        test.add(new Word("的"));
        test.add(new Word("计划"));
        LOGGER.info(recognize(test).toString());
        
        test = new ArrayList<>();
        test.add(new Word("杨尚川"));
        test.add(new Word("好"));
        LOGGER.info(recognize(test).toString());
        
        LOGGER.info(getSurname("欧阳锋"));
        LOGGER.info(getSurname("李阳锋"));
    }
}