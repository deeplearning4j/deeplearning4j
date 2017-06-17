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

import java.util.List;

/**
 * 资源加载接口
 * @author 杨尚川
 */
public interface ResourceLoader {
    /**
     * 清空数据
     */
    public void clear();
    /**
     * 初始加载全部数据
     * @param lines 初始全部数据
     */
    public void load(List<String> lines);
    /**
     * 动态增加一行数据
     * @param line 动态新增数据（一行）
     */
    public void add(String line);
    /**
     * 动态移除一行数据
     * @param line 动态移除数据（一行）
     */
    public void remove(String line);
}
