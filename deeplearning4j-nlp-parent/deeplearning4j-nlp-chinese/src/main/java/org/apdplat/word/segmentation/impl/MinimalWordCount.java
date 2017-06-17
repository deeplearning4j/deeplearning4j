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
import org.apdplat.word.recognition.RecognitionTool;
import org.apdplat.word.segmentation.Segmentation;
import org.apdplat.word.segmentation.SegmentationAlgorithm;
import org.apdplat.word.segmentation.Word;

import java.util.*;

/**
 * 最少词数算法
 * Dictionary-based minimal word count segmentation algorithm
 * 最少词数算法是指从切分结果里面选择切分出来的词的个数最少的结果
 * 如果有多个切分结果的词的个数相同
 * 则利用ngram给这些切分结果计算分值
 * 接着按分值从大到小排序
 * 最后选择第一个结果
 * @author 杨尚川
 */
public class MinimalWordCount extends AbstractSegmentation{

    @Override
    public SegmentationAlgorithm getSegmentationAlgorithm() {
        return SegmentationAlgorithm.MinimalWordCount;
    }
    @Override
    public List<Word> segImpl(String text) {
        //文本长度
        final int textLen = text.length();
        //开始虚拟节点，注意值的长度只能为1
        Node start = new Node("S", 0);
        //首节点分值为1，每两个词之间的距离都为1
        //求最短路径也就是求最少词数
        start.score = 1F;
        //结束虚拟节点
        Node end = new Node("END", textLen+1);
        //以文本中每一个字的位置（从1开始）作为二维数组的横坐标
        //以每一个字开始所能切分出来的所有的词的顺序作为纵坐标（从0开始）
        Node[][] dag = new Node[textLen+2][0];
        dag[0] = new Node[] { start };
        dag[textLen+1] = new Node[] { end };
        if(isParallelSeg()){
            //并行化
            List<Integer> list = new ArrayList<>(textLen);
            for(int i=0; i<textLen; i++){
                list.add(i);
            }
            list.parallelStream().forEach(i->dag[i+1] = fullSeg(text, i));
        }else {
            //串行化
            for (int i = 0; i < textLen; i++) {
                dag[i + 1] = fullSeg(text, i);
            }
        }
        dumpDAG(dag);
        //标注路径
        int following = 0;
        Node node = null;
        for (int i = 0; i < dag.length - 1; i++) {
            for (int j = 0; j < dag[i].length; j++) {
                node = dag[i][j];
                following = node.getFollowing();
                for (int k = 0; k < dag[following].length; k++) {
                    dag[following][k].setPrevious(node);
                }
            }
        }
        dumpShortestPath(dag);
        return toWords(end);
    }

    /**
     * 反向遍历生成分词结果
     * @param node 结束虚拟节点
     * @return 分词结果
     */
    private List<Word> toWords(Node node){
        Stack<String> stack = new Stack<>();
        while ((node = node.getPrevious()) != null) {
            if(!"S".equals(node.getText())) {
                stack.push(node.getText());
            }
        }
        int len = stack.size();
        List<Word> list = new ArrayList<>(len);
        for(int i=0; i<len; i++){
            list.add(new Word(stack.pop()));
        }
        return list;
    }

    /**
     * 获取以某个字符开始的小于截取长度的所有词
     * @param text 文本
     * @param start 起始字符索引
     * @return 所有符合要求的词
     */
    private Node[] fullSeg(final String text, final int start) {
        List<Node> result = new LinkedList<>();
        //增加单字词
        result.add(new Node(text.substring(start, start + 1), start+1));
        //文本长度
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
                result.add(new Node(text.substring(start, start + len), start+1));
            }
            len--;
        }
        return result.toArray(new Node[0]);
    }

    /**
     * 输出有向无环图的最佳路径
     * @param dag
     */
    private void dumpShortestPath(Node[][] dag){
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("有向无环图的最佳路径：");
            for (Node[] nodes : dag) {
                StringBuilder line = new StringBuilder();
                for (Node node : nodes) {
                    line.append("【")
                            .append(node.getText())
                            .append("(").append(node.getScore()).append(")")
                            .append("<-").append(node.getPrevious()==null?"":node.getPrevious().getText())
                            .append("】\t");
                }
                LOGGER.debug(line.toString());
            }
        }
    }

    /**
     * 输出有向无环图的结构
     * @param dag
     */
    private void dumpDAG(Node[][] dag){
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("有向无环图：");
            for (int i=0; i<dag.length-1; i++) {
                Node[] nodes = dag[i];
                StringBuilder line = new StringBuilder();
                for (Node node : nodes) {
                    int following = node.getFollowing();
                    StringBuilder followingNodeTexts = new StringBuilder();
                    for (int k = 0; k < dag[following].length; k++) {
                        String followingNodeText = dag[following][k].getText();
                        followingNodeTexts.append("(").append(followingNodeText).append(")");
                    }
                    line.append("【")
                            .append(node.getText())
                            .append("->").append(followingNodeTexts.toString())
                            .append("】\t");
                }
                LOGGER.debug(line.toString());
            }
        }
    }

    /**
     * 有向无环图 的 图节点
     */
    private static class Node {
        private String text;
        private Node previous;
        private int offset;
        private Float score;

        public Node(String text, int offset) {
            this.text = text;
            this.offset = offset;
        }

        public String getText() {
            return text;
        }

        public void setText(String text) {
            this.text = text;
        }

        public int getOffset() {
            return offset;
        }

        public void setOffset(int offset) {
            this.offset = offset;
        }

        public Float getScore() {
            return score;
        }

        public void setScore(Float score) {
            this.score = score;
        }

        public Node getPrevious() {
            return previous;
        }

        /**
         * 求最短路径
         * 前一个节点到当前节点的距离默认为1
         * 如果前一个节点到当前节点有ngram分值
         * 则缩短前一个节点到当前节点的距离
         * 可以放心的是前一个节点到当前节点的ngram分值大小区间为[0,1]
         * @param previous 前一个节点
         */
        public void setPrevious(Node previous) {
            float distance = 1 - Bigram.getScore(previous.getText(), this.getText());
            if (this.score == null) {
                this.score = previous.score + distance;
                this.previous = previous;
            } else if (previous.score + distance < this.score) {
                //发现更短的路径
                this.score = previous.score + distance;
                this.previous = previous;
            }
        }

        public int getFollowing() {
            return this.offset + text.length();
        }

        @Override
        public String toString() {
            return "Node{" +
                    "text='" + text + '\'' +
                    ", previous=" + previous +
                    ", offset=" + offset +
                    ", score=" + score +
                    '}';
        }
    }

    public static void main(String[] args){
        Segmentation segmentation = new MinimalWordCount();
        if(args !=null && args.length > 0){
            System.out.println(segmentation.seg(Arrays.asList(args).toString()));
            return;
        }
        System.out.println(segmentation.seg("独立自主和平等互利的原则"));
        System.out.println(segmentation.seg("我爱楚离陌"));
    }
}