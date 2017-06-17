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

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apdplat.word.util.AutoDetector;
import org.apdplat.word.util.ResourceLoader;
import org.apdplat.word.util.WordConfTools;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 数量词识别
 * @author 杨尚川
 */
public class Quantifier {
    private static final Logger LOGGER = LoggerFactory.getLogger(Quantifier.class);
    private static final Set<Character> quantifiers=new HashSet<>();
    static{
        reload();
    }
    public static void reload(){
        AutoDetector.loadAndWatch(new ResourceLoader(){

            @Override
            public void clear() {
                quantifiers.clear();
            }

            @Override
            public void load(List<String> lines) {
                LOGGER.info("初始化数量词");
                for(String line : lines){
                    if(line.length() == 1){
                        char _char = line.charAt(0);
                        if(quantifiers.contains(_char)){
                            LOGGER.info("配置文件有重复项："+line);
                        }else{
                            quantifiers.add(_char);
                        }
                    }else{
                        LOGGER.info("忽略不合法数量词："+line);
                    }
                }
                LOGGER.info("数量词初始化完毕，数量词个数："+quantifiers.size());
            }

            @Override
            public void add(String line) {
                if (line.length() == 1) {
                    char _char = line.charAt(0);
                    quantifiers.add(_char);
                } else {
                    LOGGER.info("忽略不合法数量词：" + line);
                }
            }

            @Override
            public void remove(String line) {
                if (line.length() == 1) {
                    char _char = line.charAt(0);
                    quantifiers.remove(_char);
                } else {
                    LOGGER.info("忽略不合法数量词：" + line);
                }
            }
        
        }, WordConfTools.get("quantifier.path", "classpath:quantifier.txt"));
    }
    public static boolean is(char _char){
        return quantifiers.contains(_char);
    }
    public static void main(String[] args){
        int i=1;
        for(char quantifier : quantifiers){
            LOGGER.info((i++)+" : "+quantifier);
        }
    }
}
