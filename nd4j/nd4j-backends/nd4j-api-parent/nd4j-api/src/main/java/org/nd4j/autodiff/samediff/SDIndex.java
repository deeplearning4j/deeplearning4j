/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.autodiff.samediff;
import lombok.Getter;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;

@Getter
public class SDIndex {

    public enum IndexType{
      ALL,
      POINT,
      INTERVAL
    }

    private IndexType indexType = IndexType.ALL;
    private  long pointIndex;
    private Long intervalBegin = null;
    private Long intervalEnd = null;
    private Long intervalStrides = 1l;


    public SDIndex(){}
    
    public static SDIndex all(){
        return new SDIndex();
    }
    

    public static SDIndex point(long i){
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.POINT;
        sdIndex.pointIndex = i;
        return sdIndex;
    }

    public static SDIndex interval(Long begin, Long end){
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        sdIndex.intervalBegin = begin;
        sdIndex.intervalEnd = end;
        return sdIndex;
    }

    public static SDIndex interval(Integer begin, Integer end){
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        if(begin != null) {
            sdIndex.intervalBegin = begin.longValue();
        }
        if(end != null){
            sdIndex.intervalEnd = end.longValue();
        }
        return sdIndex;
    }

    public static SDIndex interval(Long begin, Long strides, Long end){
        if(strides == 0){
            throw new ND4JIllegalArgumentException("Invalid index : strides can not be 0.");
        }
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        sdIndex.intervalBegin = begin;
        sdIndex.intervalEnd = end;
        sdIndex.intervalStrides = strides;
        return sdIndex;
    }

    public static SDIndex interval(Integer begin, Integer strides, Integer end){
        if(strides == 0){
            throw new ND4JIllegalArgumentException("Invalid index : strides can not be 0.");
        }
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        if(begin != null) {
            sdIndex.intervalBegin = begin.longValue();
        }
        if(end != null){
            sdIndex.intervalEnd = end.longValue();
        }
        if(strides != null){
            sdIndex.intervalStrides = strides.longValue();
        }
        return sdIndex;
    }
    
}
