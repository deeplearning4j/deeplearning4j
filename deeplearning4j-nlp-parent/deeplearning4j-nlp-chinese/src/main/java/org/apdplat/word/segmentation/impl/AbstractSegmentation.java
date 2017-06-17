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

package org.apdplat.word.segmentation.impl;

import org.apdplat.word.corpus.Bigram;
import org.apdplat.word.corpus.Trigram;
import org.apdplat.word.dictionary.Dictionary;
import org.apdplat.word.dictionary.DictionaryFactory;
import org.apdplat.word.recognition.PersonName;
import org.apdplat.word.recognition.Punctuation;
import org.apdplat.word.segmentation.DictionaryBasedSegmentation;
import org.apdplat.word.segmentation.Segmentation;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.Word;
import org.apdplat.word.util.WordConfTools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 基于词典的分词算法抽象类
 * @author 杨尚川
 */
public abstract class AbstractSegmentation  implements DictionaryBasedSegmentation {
    protected final Logger LOGGER = LoggerFactory.getLogger(getClass());

    private static final boolean PERSON_NAME_RECOGNIZE = WordConfTools.getBoolean("person.name.recognize", true);
    private static final boolean KEEP_WHITESPACE = WordConfTools.getBoolean("keep.whitespace", false);
    private static final boolean KEEP_PUNCTUATION = WordConfTools.getBoolean("keep.punctuation", false);
    private static final boolean PARALLEL_SEG = WordConfTools.getBoolean("parallel.seg", true);
    private static final int INTERCEPT_LENGTH = WordConfTools.getInt("intercept.length", 16);
    private static final String NGRAM = WordConfTools.get("ngram", "bigram");
    //允许动态更改词典操作接口实现
    private static Dictionary dictionary = DictionaryFactory.getDictionary();

    public boolean isParallelSeg(){
        return PARALLEL_SEG;
    }
    /**
     * 为基于词典的中文分词接口指定词典操作接口
     * @param dictionary 词典操作接口
     */
    public void setDictionary(Dictionary dictionary){
        this.dictionary.clear();
        this.dictionary = dictionary;
    }
    /**
     * 获取词典操作接口
     * @return 词典操作接口
     */
    public Dictionary getDictionary(){
        return dictionary;
    }
    /**
     * 具体的分词实现，留待子类实现
     * @param text 文本
     * @return 分词结果
     */
    public abstract List<Word> segImpl(String text);
    /**
     * 是否启用ngram
     * @return 是或否
     */
    public boolean ngramEnabled(){
        return "bigram".equals(NGRAM) || "trigram".equals(NGRAM);
    }
    /**
     * 利用ngram进行评分
     * @param sentences 多个分词结果
     * @return 评分后的结果
     */
    public Map<List<Word>, Float> ngram(List<Word>... sentences){
        if("bigram".equals(NGRAM)){
            return Bigram.bigram(sentences);
      
        }
        if("trigram".equals(NGRAM)){
            return Trigram.trigram(sentences);
        }
        return null;
    }
    /**
     * 分词时截取的字符串的最大长度
     * @return 
     */
    public int getInterceptLength(){
        if(getDictionary().getMaxLength() > INTERCEPT_LENGTH){
            return getDictionary().getMaxLength();
        }
        return INTERCEPT_LENGTH;
    }
    /**
     * 默认分词算法实现：
     * 1、把要分词的文本根据标点符号进行分割
     * 2、对分割后的文本进行分词
     * 3、组合分词结果
     * @param text 文本
     * @return 分词结果
     */
    @Override
    public List<Word> seg(String text) {
        List<String> sentences = Punctuation.seg(text, KEEP_PUNCTUATION);
        if(sentences.size() == 1){
            return segSentence(sentences.get(0));
        }
        if(!PARALLEL_SEG){
            //串行顺序处理，不能利用多核优势
            return sentences.stream().flatMap(sentence->segSentence(sentence).stream()).collect(Collectors.toList());
        }
        //如果是多个句子，可以利用多核提升分词速度
        Map<Integer, String> sentenceMap = new HashMap<>();
        int len = sentences.size();
        for(int i=0; i<len; i++){
            //记住句子的先后顺序，因为后面的parallelStream方法不保证顺序
            sentenceMap.put(i, sentences.get(i));
        }
        //用数组收集句子分词结果
        List<Word>[] results = new List[sentences.size()];
        //使用Java8中内置的并行处理机制
        sentenceMap.entrySet().parallelStream().forEach(entry -> {
            int index = entry.getKey();
            String sentence = entry.getValue();
            results[index] = segSentence(sentence);
        });
        sentences.clear();
        sentences = null;
        sentenceMap.clear();
        sentenceMap = null;
        List<Word> resultList = new ArrayList<>();
        for(List<Word> result : results){
            resultList.addAll(result);
        }
        return resultList;
    }
    /**
     * 将句子切分为词
     * @param sentence 句子
     * @return 词集合
     */
    private List<Word> segSentence(final String sentence){
        if(sentence.length() == 1){
            if(KEEP_WHITESPACE){
                List<Word> result = new ArrayList<>(1);
                result.add(new Word(sentence));
                return result;
            }else{
                if(!Character.isWhitespace(sentence.charAt(0))){
                    List<Word> result = new ArrayList<>(1);
                    result.add(new Word(sentence));
                    return result;
                }
            }
        }
        if(sentence.length() > 1){
            List<Word> list = segImpl(sentence);
            if(list != null){
                if(PERSON_NAME_RECOGNIZE){
                    list = PersonName.recognize(list);
                }
                return list;
            }else{
                LOGGER.error("文本 "+sentence+" 没有获得分词结果");
            }
        }
        return null;
    }
    /**
     * 将识别出的词放入队列
     * @param result 队列
     * @param text 文本
     * @param start 词开始索引
     * @param len 词长度
     */
    protected void addWord(List<Word> result, String text, int start, int len){
        Word word = getWord(text, start, len);
        if(word != null){
            result.add(word);
        }
    }
    /**
     * 将识别出的词入栈
     * @param result 栈
     * @param text 文本
     * @param start 词开始索引
     * @param len 词长度
     */
    protected void addWord(Stack<Word> result, String text, int start, int len){
        Word word = getWord(text, start, len);
        if(word != null){
            result.push(word);
        }
    }    
    /**
     * 获取一个已经识别的词
     * @param text 文本
     * @param start 词开始索引
     * @param len 词长度
     * @return 词或空
     */
    protected Word getWord(String text, int start, int len){
        Word word = new Word(text.substring(start, start+len).toLowerCase());
        //方便编译器优化
        if(KEEP_WHITESPACE){
            //保留空白字符
            return word;
        }else{
            //忽略空白字符
            if(len > 1){
                //长度大于1，不会是空白字符
                return word;
            }else{
                //长度为1，只要非空白字符
                if(!Character.isWhitespace(text.charAt(start))){
                    //不是空白字符，保留
                    return word;           
                }
            }
        }
        return null;
    }

    public static void main(String[] args){
        Segmentation englishSegmentation = new AbstractSegmentation() {
            @Override
            public List<Word> segImpl(String text) {
                List<Word> words = new ArrayList<>();
                for(String word : text.split("\\s+")){
                    words.add(new Word(word));
                }
                return words;
            }

            @Override
            public SegmentationAlgorithm getSegmentationAlgorithm() {
                return null;
            }
        };
        System.out.println(englishSegmentation.seg("i love programming"));
    }
}
