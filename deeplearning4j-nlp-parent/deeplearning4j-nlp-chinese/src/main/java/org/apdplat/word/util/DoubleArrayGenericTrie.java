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


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 通用的双数组前缀树的Java实现
 * 用于快速检索 K V 对
 * An Implementation of Double-Array Trie: http://linux.thai.net/~thep/datrie/datrie.html
 * @author 杨尚川
 */
public class DoubleArrayGenericTrie{
    private static final Logger LOGGER = LoggerFactory.getLogger(DoubleArrayGenericTrie.class);
    private int size = 65000;

    public DoubleArrayGenericTrie(int size){
        this.size = size;
    }

    private static class Node {
        private int code;
        private int depth;
        private int left;
        private int right;
        private int value;

        @Override
        public String toString() {
            return "Node{" +
                    "code=" + code + "["+ (char)code + "]" +
                    ", depth=" + depth +
                    ", left=" + left +
                    ", right=" + right +
                    ", value=" + value +
                    '}';
        }
    };

    private int[] check;
    private int[] base;
    private boolean[] used;
    private int nextCheckPos;

    public DoubleArrayGenericTrie(){
        LOGGER.info("初始化双数组前缀树：" + this.getClass().getName());
    }

    private List<Node> toTree(Node parent, List<String> items, Map<String, Integer> map) {
        List<Node> siblings = new ArrayList<>();
        int prev = 0;

        for (int i = parent.left; i < parent.right; i++) {
            if (items.get(i).length() < parent.depth)
                continue;

            String item = items.get(i);

            int cur = 0;
            if (item.length() != parent.depth) {
                cur = (int) item.charAt(parent.depth);
            }

            if (cur != prev || siblings.isEmpty()) {
                Node node = new Node();
                node.depth = parent.depth + 1;
                node.code = cur;
                node.left = i;
                if(cur==0 || cur==item.charAt(item.length()-1)){
                    if(map.get(item)!=null) {
                        node.value = map.get(item);
                    }
                }
                if (!siblings.isEmpty()) {
                    siblings.get(siblings.size() - 1).right = i;
                }
                siblings.add(node);
            }

            prev = cur;
        }

        if (!siblings.isEmpty()) {
            siblings.get(siblings.size() - 1).right = parent.right;
            if(LOGGER.isDebugEnabled()) {
                if (items.size()<10) {
                    LOGGER.debug("************************************************");
                    LOGGER.debug("树信息：");
                    siblings.forEach(s -> LOGGER.debug(s.toString()));
                    LOGGER.debug("************************************************");
                }
            }
        }
        return siblings;
    }

    private int toDoubleArray(List<Node> siblings, List<String> words, Map<String, Integer> map) {
        int begin = 0;
        int index = (siblings.get(0).code > nextCheckPos) ? siblings.get(0).code : nextCheckPos;
        boolean isFirst = true;

        outer: while (true) {
            index++;

            if (check[index] != 0) {
                continue;
            } else if (isFirst) {
                nextCheckPos = index;
                isFirst = false;
            }

            begin = index - siblings.get(0).code;

            if (used[begin]) {
                continue;
            }

            for (int i = 1; i < siblings.size(); i++) {
                if (check[begin + siblings.get(i).code] != 0) {
                    continue outer;
                }
            }

            break;
        }

        used[begin] = true;

        for (int i = 0; i < siblings.size(); i++) {
            check[begin + siblings.get(i).code] = begin;
        }

        for (int i = 0; i < siblings.size(); i++) {
            List<Node> newSiblings = toTree(siblings.get(i), words, map);

            if (newSiblings.isEmpty()) {
                base[begin + siblings.get(i).code] = -1;
                check[begin + siblings.get(i).code] = siblings.get(i).value;
            } else {
                int h = toDoubleArray(newSiblings, words, map);
                base[begin + siblings.get(i).code] = h;
            }
        }
        return begin;
    }
    private void allocate(int size){
        check = null;
        base = null;
        used = null;
        nextCheckPos = 0;

        base = new int[size];
        check = new int[size];
        used = new boolean[size];
        base[0] = 1;
    }
    private void init(List<String> items, Map<String, Integer> map) {
        if (items == null || items.isEmpty()) {
            return;
        }

        //前缀树的虚拟根节点
        Node rootNode = new Node();
        rootNode.left = 0;
        rootNode.right = items.size();
        rootNode.depth = 0;

        while (true) {
            try {
                allocate(size);
                List<Node> siblings = toTree(rootNode, items, map);
                toDoubleArray(siblings, items, map);
                break;
            } catch (Exception e) {
                size += size/10;
                LOGGER.error("分配空间不够，增加至： " + size);
            }
        }

        items.clear();
        items = null;
        map.clear();
        map=null;
        used = null;
    }

    public int get(String item, int start, int length) {
        if(LOGGER.isDebugEnabled()){
            LOGGER.debug("开始查询数据：{}", item.substring(start, start + length));
        }
        if(base==null){
            return Integer.MIN_VALUE;
        }

        //base[0]=1
        int lastChar = base[0];
        int index;

        for (int i = start; i < start+length; i++) {
            index = lastChar + (int) item.charAt(i);
            if(index >= check.length || index < 0){
                return Integer.MIN_VALUE;
            }
            if (lastChar == check[index]) {
                lastChar = base[index];
            }else {
                return Integer.MIN_VALUE;
            }
        }
        index = lastChar;
        if(index >= check.length || index < 0){
            return Integer.MIN_VALUE;
        }
        if (base[index] < 0) {
            if(LOGGER.isDebugEnabled()) {
                LOGGER.debug("在词典中查到词：{}", item.substring(start, start + length));
            }
            return check[lastChar];
        }
        return Integer.MIN_VALUE;
    }

    public int get(String item) {
        return get(item, 0, item.length());
    }

    public void putAll(Map<String, Integer> map) {
        if(check!=null){
            throw new RuntimeException("addAll method can just be used once after clear method!");
        }

        List<String> items=map
                            .keySet()
                            .stream()
                            .sorted()
                            .collect(Collectors.toList());
        if(LOGGER.isDebugEnabled()){
            //for debug
            if (items.size()<10){
                items.forEach(item->LOGGER.debug(item+"="+map.get(item)));
            }
        }
        init(items, map);
    }

    public void clear() {
        check = null;
        base = null;
        used = null;
        nextCheckPos = 0;
    }

    public static void main(String[] args) {
        DoubleArrayGenericTrie doubleArrayGenericTrie = new DoubleArrayGenericTrie();

        Map<String, Integer> map = new HashMap<>();
        map.put("杨尚川", 100);
        map.put("章子怡", 101);
        map.put("刘亦菲", 99);
        map.put("刘", 11);
        map.put("刘诗诗", -1);
        map.put("巩俐", 1);
        map.put("中国", 2);
        map.put("主演", 3);

        //构造双数组前缀树
        doubleArrayGenericTrie.putAll(map);
        System.out.println("增加数据");

        System.out.println("查找 杨尚川：" + doubleArrayGenericTrie.get("杨尚川"));
        System.out.println("查找 章子怡：" + doubleArrayGenericTrie.get("章子怡"));
        System.out.println("查找 刘："+doubleArrayGenericTrie.get("刘"));
        System.out.println("查找 刘亦菲：" + doubleArrayGenericTrie.get("刘亦菲"));
        System.out.println("查找 刘诗诗：" + doubleArrayGenericTrie.get("刘诗诗"));
        System.out.println("查找 巩俐："+doubleArrayGenericTrie.get("巩俐"));
        System.out.println("查找 中国的巩俐是红高粱的主演 3 2：" + doubleArrayGenericTrie.get("中国的巩俐是红高粱的主演", 3, 2));
        System.out.println("查找 中国的巩俐是红高粱的主演 0 2：" + doubleArrayGenericTrie.get("中国的巩俐是红高粱的主演", 0, 2));
        System.out.println("查找 中国的巩俐是红高粱的主演 10 2：" + doubleArrayGenericTrie.get("中国的巩俐是红高粱的主演", 10, 2));
        System.out.println("查找 复仇者联盟2：" + doubleArrayGenericTrie.get("复仇者联盟2"));
        System.out.println("查找 白掌：" + doubleArrayGenericTrie.get("白掌"));
        System.out.println("查找 红掌：" + doubleArrayGenericTrie.get("红掌"));

        doubleArrayGenericTrie.clear();
        System.out.println("清除所有数据");

        System.out.println("查找 杨尚川：" + doubleArrayGenericTrie.get("杨尚川"));
        System.out.println("查找 章子怡：" + doubleArrayGenericTrie.get("章子怡"));

        map.put("白掌", 1000);
        map.put("红掌", 1001);
        map.put("复仇者联盟2", -1000);

        doubleArrayGenericTrie.putAll(map);
        System.out.println("增加数据");

        System.out.println("查找 杨尚川：" + doubleArrayGenericTrie.get("杨尚川"));
        System.out.println("查找 章子怡：" + doubleArrayGenericTrie.get("章子怡"));
        System.out.println("查找 复仇者联盟2："+doubleArrayGenericTrie.get("复仇者联盟2"));
        System.out.println("查找 白掌：" + doubleArrayGenericTrie.get("白掌"));
        System.out.println("查找 红掌："+doubleArrayGenericTrie.get("红掌"));
        System.out.println("查找 刘亦菲："+doubleArrayGenericTrie.get("刘亦菲"));
        System.out.println("查找 刘诗诗："+doubleArrayGenericTrie.get("刘诗诗"));
        System.out.println("查找 巩俐：" + doubleArrayGenericTrie.get("巩俐"));
        System.out.println("查找 金钱树："+doubleArrayGenericTrie.get("金钱树"));
    }
}