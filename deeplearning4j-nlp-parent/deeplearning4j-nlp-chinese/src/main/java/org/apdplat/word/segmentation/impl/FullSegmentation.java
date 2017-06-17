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

import java.util.*;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.apdplat.word.recognition.RecognitionTool;
import org.apdplat.word.segmentation.Segmentation;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.SegmentationFactory;
import org.apdplat.word.segmentation.Word;

/**
 * 基于词典的全切分算法
 * Dictionary-based full segmentation algorithm
 * 利用ngram给每一种切分结果计算分值
 * 如果多个切分结果分值相同，则选择切分出的词的个数最少的切分结果（最少分词原则）
 * @author 杨尚川
 */
public class FullSegmentation extends AbstractSegmentation{
    private static final AbstractSegmentation RMM = (AbstractSegmentation)SegmentationFactory.getSegmentation(SegmentationAlgorithm.ReverseMaximumMatching);
    //在评估采用的测试文本253 3709行2837 4490个字符中,行长度小于等于50的占了99.932465%
    //大于50的行长度文本采用逆向最大匹配算法切分文本
    private static final int PROCESS_TEXT_LENGTH_LESS_THAN = 50;
    //长度小于等于18的文本单字成词，大于18的文本只有无词时才单字成词
    private static final int CHAR_IS_WORD_LENGTH_LESS_THAN = 18;

    @Override
    public SegmentationAlgorithm getSegmentationAlgorithm() {
        return SegmentationAlgorithm.FullSegmentation;
    }
    @Override
    public List<Word> segImpl(String text) {
        if(text.length() > PROCESS_TEXT_LENGTH_LESS_THAN){
            return RMM.segImpl(text);
        }
        //获取全切分结果
        List<Word>[] array = fullSeg(text);
        //利用ngram计算分值
        Map<List<Word>, Float> words = ngram(array);
        //歧义消解（ngram分值优先、词个数少优先）
        List<Word> result = disambiguity(words);
        return result;
    }
    private List<Word> disambiguity(Map<List<Word>, Float> words){
        //按分值排序
        List<Entry<List<Word>, Float>> entrys = words.entrySet().parallelStream().sorted((a,b)->b.getValue().compareTo(a.getValue())).collect(Collectors.toList());
        if(LOGGER.isDebugEnabled()){
            LOGGER.debug("ngram分值：");
            int i=1;
            for(Entry<List<Word>, Float> entry : entrys){
                LOGGER.debug("\t"+(i++)+"、"+"词个数="+entry.getKey().size()+"\tngram分值="+entry.getValue()+"\t"+entry.getKey());
            }
        }
        //移除小于最大分值的切分结果
        float maxScore=entrys.get(0).getValue();
        Iterator<Entry<List<Word>, Float>> iter = entrys.iterator();
        while(iter.hasNext()){
            Entry<List<Word>, Float> entry = iter.next();
            if(entry.getValue() < maxScore){
                entry.getKey().clear();
                iter.remove();
            }
        }
        if(LOGGER.isDebugEnabled()){
            LOGGER.debug("只保留最大分值：");
            int i=1;
            for(Entry<List<Word>, Float> entry : entrys){
                LOGGER.debug("\t"+(i++)+"、"+"词个数="+entry.getKey().size()+"\tngram分值="+entry.getValue()+"\t"+entry.getKey());
            }
        }
        //如果有多个分值一样的切分结果，则选择词个数最少的（最少分词原则）
        int minSize=Integer.MAX_VALUE;
        List<Word> minSizeList = null;
        iter = entrys.iterator();
        while(iter.hasNext()){
            Entry<List<Word>, Float> entry = iter.next();
            if(entry.getKey().size() < minSize){
                minSize = entry.getKey().size();
                if(minSizeList != null){
                    minSizeList.clear();
                }
                minSizeList = entry.getKey();
            }else{
                entry.getKey().clear();
                iter.remove();
            }
        }
        if(LOGGER.isDebugEnabled()){
            LOGGER.debug("最大分值："+maxScore+", 消歧结果："+minSizeList+"，词个数："+minSize);
        }
        return minSizeList;
    }
    /**
     * 获取文本的所有可能切分结果
     * @param text 文本
     * @return 全切分结果
     */
    private List<Word>[] fullSeg(String text){
        //文本长度
        final int textLen = text.length();
        //以每一个字作为词的开始，所能切分的词
        final List<String>[] sequence = new LinkedList[textLen];
        if(isParallelSeg()){
            //并行化
            List<Integer> list = new ArrayList<>(textLen);
            for(int i=0; i<textLen; i++){
                list.add(i);
            }
            list.parallelStream().forEach(i->sequence[i] = fullSeg(text, i));
        }else {
            //串行化
            for (int i = 0; i < textLen; i++) {
                sequence[i] = fullSeg(text, i);
            }
        }
        if(LOGGER.isDebugEnabled()){
            LOGGER.debug("全切分中间结果：");
            int i=1;
            for(List<String> list : sequence){
                LOGGER.debug("\t"+(i++)+"、"+list);
            }
        }
        //树叶
        List<Node> leaf = new LinkedList<>();
        for(String word : sequence[0]){
            //树根
            Node node = new Node(word);
            //把全切分中间结果（二维数组）转换为合理切分
            buildNode(node, sequence, word.length(), leaf);
        }
        //清理无用数据
        for(int j=0; j<sequence.length; j++){
            sequence[j].clear();
            sequence[j] = null;
        }
        //从所有树叶开始反向遍历出全切分结果
        List<Word>[] res = toWords(leaf);
        leaf.clear();
        return res;
    }
    /**
     * 获取以某个字符开始的小于截取长度的所有词
     * @param text 文本
     * @param start 起始字符索引
     * @return 所有符合要求的词
     */
    private List<String> fullSeg(final String text, final int start) {
        List<String> result = new LinkedList<>();
        final int textLen = text.length();
        //剩下文本长度
        int len = textLen - start;
        //最大截取长度
        int interceptLength = getInterceptLength();
        if(len > interceptLength){
            len = interceptLength;
        }
        while(len > 1){
            if(getDictionary().contains(text, start, len) || RecognitionTool.recog(text, start, len)){
                result.add(text.substring(start, start + len));
            }
            len--;
        }
        if(textLen <= CHAR_IS_WORD_LENGTH_LESS_THAN || result.isEmpty()){
            //增加单字词
            result.add(text.substring(start, start + 1));
        }
        return result;
    }
    /**
     * 根据全切分中间结果构造切分树
     * @param parent 父节点
     * @param sequence 全切分中间结果
     * @param from 全切分中间结果数组下标索引
     * @param leaf 叶子节点集合
     */
    private void buildNode(Node parent, List<String>[] sequence, int from, List<Node> leaf){
        //递归退出条件：二维数组遍历完毕
        if(from >= sequence.length){
            //记住叶子节点
            leaf.add(parent);
            return;
        }
        for(String item : sequence[from]){
            Node child = new Node(item, parent);
            buildNode(child, sequence, from+item.length(), leaf);
        }
    }
    /**
     * 从树叶开始反向遍历生成全切分结果
     * @param leaf 树叶节点集合
     * @return 全切分结果集合
     */
    private List<Word>[] toWords(List<Node> leaf){
        List<Word>[] result = new ArrayList[leaf.size()];
        int i = 0;
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("全切分结果：");
        }
        for(Node node : leaf){
            result[i++] = toWords(node);
            if(LOGGER.isDebugEnabled()) {
                LOGGER.debug("\t" + i + "：" + result[i - 1]);
            }
        }
        return result;
    }
    /**
     * 从树叶开始反向遍历生成全切分结果
     * @param node 树叶节点
     * @return 全切分结果
     */
    private List<Word> toWords(Node node){
        Stack<String> stack = new Stack<>();
        while(node != null){
            stack.push(node.getText());
            node = node.getParent();
        }
        int len = stack.size();
        List<Word> list = new ArrayList<>(len);
        for(int i=0; i<len; i++){
            list.add(new Word(stack.pop()));
        }
        return list;
    }
    /**
     * 树节点
     * 只需要反向遍历
     * 不需要记住子节点，知道父节点即可
     */
    private static class Node{
        private String text;
        private Node parent;
        public Node(String text) {
            this.text = text;
        }
        public Node(String text, Node parent) {
            this.text = text;
            this.parent = parent;
        }
        public String getText() {
            return text;
        }
        public void setText(String text) {
            this.text = text;
        }
        public Node getParent() {
            return parent;
        }
        public void setParent(Node parent) {
            this.parent = parent;
        }
        @Override
        public String toString() {
            return this.text;
        }
    }
    public static void main(String[] args){
        Segmentation m = new FullSegmentation();
        if(args !=null && args.length > 0){
            System.out.println(m.seg(Arrays.asList(args).toString()));
            return;
        }
        String text = "蝶舞打扮得漂漂亮亮出现在张公公面前";
        System.out.println(m.seg(text));
    }
}