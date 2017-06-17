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

package org.apdplat.word.analysis;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 相似度排名结果列表
 * @author 杨尚川
 */
public class Hits {
    private List<Hit> hits = null;

    public Hits(){
        hits = new ArrayList<>();
    }

    public Hits(int size){
        hits = new ArrayList<>(size);
    }

    public int size(){
        return hits.size();
    }

    public List<Hit> getHits() {
        return Collections.unmodifiableList(hits);
    }

    public void addHits(List<Hit> hits) {
        this.hits.addAll(hits);
    }
    public void addHit(Hit hit) {
        this.hits.add(hit);
    }
}
