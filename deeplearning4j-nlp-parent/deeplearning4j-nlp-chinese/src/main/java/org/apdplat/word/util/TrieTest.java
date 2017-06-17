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

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

/**
 * 前缀树和双数组前缀树性能测试
 * @author 杨尚川
 */
public class TrieTest {
    private static final Logger LOGGER = LoggerFactory.getLogger(TrieTest.class);
    public static void testBigram() throws Exception{
        Map<String, Integer> map = new HashMap<>();
        Stream<String> lines = Files.lines(Paths.get("src/test/resources/bigram.txt"));
        lines.forEach(line -> {
            String[] attrs = line.split("\\s+");
            if(attrs!=null && attrs.length==2){
                map.put(attrs[0], Integer.parseInt(attrs[1]));
            }
        });
        DoubleArrayGenericTrie doubleArrayGenericTrie = new DoubleArrayGenericTrie(WordConfTools.getInt("bigram.double.array.trie.size", 5500000));
        doubleArrayGenericTrie.putAll(map);
        map.keySet().forEach(key->assertEquals(map.get(key).intValue(), doubleArrayGenericTrie.get(key)));
        for(int i=0; i<1000; i++){
            map.keySet().forEach(key->doubleArrayGenericTrie.get(key));
        }
    }
    public static void testTrigram() throws Exception{
        Map<String, Integer> map = new HashMap<>();
        Stream<String> lines = Files.lines(Paths.get("src/test/resources/trigram.txt"));
        lines.forEach(line -> {
            String[] attrs = line.split("\\s+");
            if(attrs!=null && attrs.length==2){
                map.put(attrs[0], Integer.parseInt(attrs[1]));
            }
        });
        DoubleArrayGenericTrie doubleArrayGenericTrie = new DoubleArrayGenericTrie(WordConfTools.getInt("trigram.double.array.trie.size", 10100000));
        doubleArrayGenericTrie.putAll(map);
        map.keySet().forEach(key->assertEquals(map.get(key).intValue(), doubleArrayGenericTrie.get(key)));
        for(int i=0; i<1000; i++){
            map.keySet().forEach(key->doubleArrayGenericTrie.get(key));
        }
    }

    public static void testBigram2() throws Exception{
        GenericTrie<Integer> genericTrie = new GenericTrie<>();
        Map<String, Integer> map = new HashMap<>();
        Stream<String> lines = Files.lines(Paths.get("src/test/resources/bigram.txt"));
        lines.forEach(line -> {
            String[] attrs = line.split("\\s+");
            if(attrs!=null && attrs.length==2){
                map.put(attrs[0], Integer.parseInt(attrs[1]));
                genericTrie.put(attrs[0], map.get(attrs[0]));
            }
        });
        map.keySet().forEach(key->assertEquals(map.get(key).intValue(), genericTrie.get(key).intValue()));
        for(int i=0; i<1000; i++){
            map.keySet().forEach(key->genericTrie.get(key));
        }
    }
    public static void testTrigram2() throws Exception{
        GenericTrie<Integer> genericTrie = new GenericTrie<>();
        Map<String, Integer> map = new HashMap<>();
        Stream<String> lines = Files.lines(Paths.get("src/test/resources/trigram.txt"));
        lines.forEach(line -> {
            String[] attrs = line.split("\\s+");
            if(attrs!=null && attrs.length==2){
                map.put(attrs[0], Integer.parseInt(attrs[1]));
                genericTrie.put(attrs[0], map.get(attrs[0]));
            }
        });
        map.keySet().forEach(key->assertEquals(map.get(key).intValue(), genericTrie.get(key).intValue()));
        for(int i=0; i<1000; i++){
            map.keySet().forEach(key->genericTrie.get(key));
        }
    }
    private static void assertEquals(int v1, int v2){
        if(v1!=v2){
            throw new RuntimeException(v1+" not equals "+v2);
        }
    }
    private static void test(String type, String trie) throws Exception{
        System.gc();
        Thread.sleep(60000);
        long start = System.currentTimeMillis();
        if("bigram".equals(type)) {
            if("dat".equals(trie)) {
                testBigram();
            }
            if("t".equals(trie)) {
                testBigram2();
            }
        }
        if("trigram".equals(type)) {
            if("dat".equals(trie)) {
                testTrigram();
            }
            if("t".equals(trie)) {
                testTrigram2();
            }
        }
        long cost = System.currentTimeMillis() - start;
        LOGGER.info(type+":"+trie+" 耗时：" + cost + " 毫秒");
        System.gc();
        Thread.sleep(60000);
        LOGGER.info("test finish");
        System.exit(0);
    }
    public static void main(String[] args) throws Exception{
        //速度和内存测试
        //打开 jvisualvm 后分别运行测试，不同测试位于不同进程中，避免相互影响
        //通过 jvisualvm 可以查看内存使用情况
        test("bigram", "t");
        //test("bigram", "dat");
        //test("trigram", "t");
        //test("trigram", "dat");
    }
}
