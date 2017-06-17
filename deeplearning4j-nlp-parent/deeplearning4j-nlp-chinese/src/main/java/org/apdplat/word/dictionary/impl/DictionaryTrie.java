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

package org.apdplat.word.dictionary.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apdplat.word.dictionary.Dictionary;
import org.apdplat.word.util.WordConfTools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 词首字索引式前缀树
 * 前缀树的Java实现
 * 为前缀树的一级节点（词首字）建立索引（比二分查找要快）
 * 用于查找一个指定的字符串是否在词典中
 * @author 杨尚川
 */
public class DictionaryTrie implements Dictionary{
    private static final Logger LOGGER = LoggerFactory.getLogger(DictionaryTrie.class);
    //词表的首字母数量在一个可控范围内，默认值为24000
    private static final int INDEX_LENGTH = WordConfTools.getInt("dictionary.trie.index.size", 24000);;
    private final TrieNode[] ROOT_NODES_INDEX = new TrieNode[INDEX_LENGTH];
    private int maxLength;
    public DictionaryTrie(){
        LOGGER.info("初始化词典："+this.getClass().getName());
    }
    @Override
    public void clear() {
        for(int i=0; i<INDEX_LENGTH; i++){
            ROOT_NODES_INDEX[i] = null;
        }
    }
    /**
     * 统计根节点冲突情况及预分配的数组空间利用情况
     */
    public void showConflict(){
        int emptySlot=0;
        //key:冲突长度 value:冲突个数
        Map<Integer, Integer> map = new HashMap<>();
        for(TrieNode node : ROOT_NODES_INDEX){
            if(node == null){
                emptySlot++;
            }else{
                int i=0;
                while((node = node.getSibling()) != null){
                    i++;
                }
                if(i > 0){
                    Integer count = map.get(i);
                    if(count == null){
                        count = 1;
                    }else{
                        count++;
                    }
                    map.put(i, count);
                }
            }
        }
        int count=0;
        for(int key : map.keySet()){
            int value = map.get(key);
            count += key*value;
            LOGGER.info("冲突次数为："+key+" 的元素个数："+value);
        }
        LOGGER.info("冲突次数："+count);
        LOGGER.info("总槽数："+INDEX_LENGTH);
        LOGGER.info("用槽数："+(INDEX_LENGTH-emptySlot));
        LOGGER.info("使用率："+(float)(INDEX_LENGTH-emptySlot)/INDEX_LENGTH*100+"%");
        LOGGER.info("剩槽数："+emptySlot);
    }
    /**
     * 获取字符对应的根节点
     * 如果节点不存在
     * 则增加根节点后返回新增的节点
     * @param character 字符
     * @return 字符对应的根节点
     */
    private TrieNode getRootNodeIfNotExistThenCreate(char character){
        TrieNode trieNode = getRootNode(character);
        if(trieNode == null){
            trieNode = new TrieNode(character);
            addRootNode(trieNode);
        }
        return trieNode;
    }
    /**
     * 新增一个根节点
     * @param rootNode 根节点
     */
    private void addRootNode(TrieNode rootNode){
        //计算节点的存储索引
        int index = rootNode.getCharacter()%INDEX_LENGTH;
        //检查索引是否和其他节点冲突
        TrieNode existTrieNode = ROOT_NODES_INDEX[index];
        if(existTrieNode != null){
            //有冲突，将冲突节点附加到当前节点之后
            rootNode.setSibling(existTrieNode);
        }
        //新增的节点总是在最前
        ROOT_NODES_INDEX[index] = rootNode;
    }
    /**
     * 获取字符对应的根节点
     * 如果不存在，则返回NULL
     * @param character 字符
     * @return 字符对应的根节点
     */
    private TrieNode getRootNode(char character){
        //计算节点的存储索引
        int index = character%INDEX_LENGTH;
        TrieNode trieNode = ROOT_NODES_INDEX[index];
        while(trieNode != null && character != trieNode.getCharacter()){
            //如果节点和其他节点冲突，则需要链式查找
            trieNode = trieNode.getSibling();
        }
        return trieNode;
    }
    public List<String> prefix(String prefix){
        List<String> result = new ArrayList<>();
        //去掉首尾空白字符
        prefix=prefix.trim();
        int len = prefix.length();    
        if(len < 1){
            return result;
        }          
        //从根节点开始查找
        //获取根节点
        TrieNode node = getRootNode(prefix.charAt(0));
        if(node == null){
            //不存在根节点，结束查找
            return result;
        }
        //存在根节点，继续查找
        for(int i=1;i<len;i++){
            char character = prefix.charAt(i);
            TrieNode child = node.getChild(character);
            if(child == null){
                //未找到匹配节点
                return result;
            }else{
                //找到节点，继续往下找
                node = child;
            }
        }
        for(TrieNode item : node.getChildren()){            
            result.add(prefix+item.getCharacter());
        }
        return result;
    }
    
    @Override
    public boolean contains(String item){
        return contains(item, 0, item.length());
    }
    @Override
    public boolean contains(String item, int start, int length){
        if(start < 0 || length < 1){
            return false;
        }
        if(item == null || item.length() < length){
            return false;
        }
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("开始查词典：{}", item.substring(start, start + length));
        }
        //从根节点开始查找
        //获取根节点
        TrieNode node = getRootNode(item.charAt(start));
        if(node == null){
            //不存在根节点，结束查找
            return false;
        }
        //存在根节点，继续查找
        for(int i=1;i<length;i++){
            char character = item.charAt(i+start);
            TrieNode child = node.getChild(character);
            if(child == null){
                //未找到匹配节点
                return false;
            }else{
                //找到节点，继续往下找
                node = child;
            }
        }
        if(node.isTerminal()){
            if(LOGGER.isDebugEnabled()) {
                LOGGER.debug("在词典中查到词：{}", item.substring(start, start + length));
            }
            return true;
        }
        return false;
    }
    
    @Override
    public void removeAll(List<String> items) {
        for(String item : items){
            remove(item);
        }
    }

    @Override
    public void remove(String item) {
        if(item == null || item.isEmpty()){
            return;
        }
        if(LOGGER.isDebugEnabled()) {
            LOGGER.debug("从词典中移除词：{}", item);
        }
        //从根节点开始查找
        //获取根节点
        TrieNode node = getRootNode(item.charAt(0));
        if(node == null){
            //不存在根节点，结束查找
            if (LOGGER.isDebugEnabled()) {
                LOGGER.debug("词不存在：{}", item);
            }
            return;
        }
        int length = item.length();
        //存在根节点，继续查找
        for(int i=1;i<length;i++){
            char character = item.charAt(i);
            TrieNode child = node.getChild(character);
            if(child == null){
                //未找到匹配节点
                if (LOGGER.isDebugEnabled()) {
                    LOGGER.debug("词不存在：{}", item);
                }
                return;
            }else{
                //找到节点，继续往下找
                node = child;
            }
        }
        if(node.isTerminal()){
            //设置为非叶子节点，效果相当于从词典中移除词
            node.setTerminal(false);
            if (LOGGER.isDebugEnabled()) {
                LOGGER.debug("成功从词典中移除词：{}", item);
            }
        }else{
            if (LOGGER.isDebugEnabled()) {
                LOGGER.debug("词不存在：{}", item);
            }
        }
    }
    
    @Override
    public void addAll(List<String> items){
        for(String item : items){
            add(item);
        }
    }
    @Override
    public void add(String item){
        //去掉首尾空白字符
        item=item.trim();
        int len = item.length();
        if(len < 1){
            //长度小于1则忽略
            return;
        }
        if(len>maxLength){
            maxLength=len;
        }
        //从根节点开始添加
        //获取根节点
        TrieNode node = getRootNodeIfNotExistThenCreate(item.charAt(0));
        for(int i=1;i<len;i++){
            char character = item.charAt(i);
            TrieNode child = node.getChildIfNotExistThenCreate(character);
            //改变顶级节点
            node = child;
        }
        //设置终结字符，表示从根节点遍历到此是一个合法的词
        node.setTerminal(true);
    }
    
    @Override
    public int getMaxLength() {
        return maxLength;
    }
    private static class TrieNode implements Comparable{
        private char character;
        private boolean terminal;
        private TrieNode sibling;
        private TrieNode[] children = new TrieNode[0];
        public TrieNode(char character){
            this.character = character;
        }
        public boolean isTerminal() {
            return terminal;
        }
        public void setTerminal(boolean terminal) {
            this.terminal = terminal;
        }        
        public char getCharacter() {
            return character;
        }
        public void setCharacter(char character) {
            this.character = character;
        }
        public TrieNode getSibling() {
            return sibling;
        }
        public void setSibling(TrieNode sibling) {
            this.sibling = sibling;
        }
        public Collection<TrieNode> getChildren() {
            return Arrays.asList(children);            
        }
        /**
         * 利用二分搜索算法从有序数组中找到特定的节点
         * @param character 待查找节点
         * @return NULL OR 节点数据
         */
        public TrieNode getChild(char character) {
            int index = Arrays.binarySearch(children, character);
            if(index >= 0){
                return children[index];
            }
            return null;
        }        
        public TrieNode getChildIfNotExistThenCreate(char character) {
            TrieNode child = getChild(character);
            if(child == null){
                child = new TrieNode(character);
                addChild(child);
            }
            return child;
        }
        public void addChild(TrieNode child) {
            children = insert(children, child);
        }
        /**
         * 将一个字符追加到有序数组
         * @param array 有序数组
         * @param element 字符
         * @return 新的有序数字
         */
        private TrieNode[] insert(TrieNode[] array, TrieNode element){
            int length = array.length;
            if(length == 0){
                array = new TrieNode[1];
                array[0] = element;
                return array;
            }
            TrieNode[] newArray = new TrieNode[length+1];
            boolean insert=false;
            for(int i=0; i<length; i++){
                if(element.getCharacter() <= array[i].getCharacter()){
                    //新元素找到合适的插入位置
                    newArray[i]=element;
                    //将array中剩下的元素依次加入newArray即可退出比较操作
                    System.arraycopy(array, i, newArray, i+1, length-i);
                    insert=true;
                    break;
                }else{
                    newArray[i]=array[i];
                }
            }
            if(!insert){
                //将新元素追加到尾部
                newArray[length]=element;
            }
            return newArray;
        }
        /**
         * 注意这里的比较对象是char
         * @param o char
         * @return 
         */
        @Override
        public int compareTo(Object o) {
            return this.getCharacter()-(char)o;
        }
    }    
    public void show(char character){
        show(getRootNode(character), "");
    }
    public void show(){
        for(TrieNode node : ROOT_NODES_INDEX){
            if(node != null){
                show(node, "");
            }
        }
    }
    private void show(TrieNode node, String indent){
        if(node.isTerminal()){
            LOGGER.info(indent+node.getCharacter()+"(T)");
        }else{
            LOGGER.info(indent+node.getCharacter());
        }        
        for(TrieNode item : node.getChildren()){
            show(item,indent+"\t");
        }
    }
    public static void main(String[] args){
        DictionaryTrie trie = new DictionaryTrie();
        trie.add("APDPlat");
        trie.add("APP");
        trie.add("APD");
        trie.add("杨尚川");
        trie.add("杨尚昆");
        trie.add("杨尚喜");
        trie.add("中华人民共和国");
        trie.add("中华人民打太极");
        trie.add("中华");
        trie.add("中心思想");
        trie.add("杨家将");        
        trie.show();
        LOGGER.info(trie.prefix("中").toString());
        LOGGER.info(trie.prefix("中华").toString());
        LOGGER.info(trie.prefix("杨").toString());
        LOGGER.info(trie.prefix("杨尚").toString());
    }
}
