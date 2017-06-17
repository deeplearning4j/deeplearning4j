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
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.concurrent.atomic.AtomicLong;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 将多个语料库文件合并为一个
 * @author 杨尚川
 */
public class CorpusMerge {
    private static final Logger LOGGER = LoggerFactory.getLogger(CorpusMerge.class);
    
    public static void main(String[] args) throws IOException{
        //注意输入语料库的文件编码要为UTF-8
        String source = "d:/corpora";
        String target = "src/main/resources/corpus/corpus.txt";
        merge(source, target);
    }
    /**
     * 将多个语料库文件合并为一个
     * @param source 目录，可多级嵌套
     * @param target 目标文件
     * @throws IOException 
     */
    public static void merge(String source, String target) throws IOException{
        final AtomicLong count = new AtomicLong();
        final AtomicLong lines = new AtomicLong();
        try(BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(target),"utf-8"))){
            Files.walkFileTree(Paths.get(source), new SimpleFileVisitor<Path>(){

                    @Override
                    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                        LOGGER.info("处理文件："+file);
                        try(BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file.toFile()),"utf-8"));){
                            String line;
                            while( (line = reader.readLine()) != null ){
                                count.addAndGet(line.length());
                                lines.incrementAndGet();
                                writer.write(line+"\n");
                            }
                        }
                        return FileVisitResult.CONTINUE;
                    }
                    
                });
        }
        LOGGER.info("语料库行数："+lines.get());
        LOGGER.info("语料库字符数目："+count.get());
    }
}
