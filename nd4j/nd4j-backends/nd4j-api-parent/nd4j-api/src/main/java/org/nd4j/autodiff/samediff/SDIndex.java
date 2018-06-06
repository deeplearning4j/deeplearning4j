package org.nd4j.autodiff.samediff;

import org.nd4j.linalg.exception.ND4JIllegalArgumentException;

public class SDIndex {

    public enum IndexType{
      ALL,
      POINT,
      INTERVAL
    }

    public IndexType indexType = IndexType.ALL;
    
    public SDIndex(){}
    
    public static SDIndex all(){
        return new SDIndex();
    }
    
    public Integer point_index;
    public static SDIndex point(Integer i){
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.POINT;
        sdIndex.point_index = i;
        return sdIndex;
    }
    
    public Integer interval_begin;
    public Integer interval_end;
    public static SDIndex interval(Integer begin, Integer end){
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        sdIndex.interval_begin = begin;
        sdIndex.interval_end = end;
        return sdIndex;
    }

    public Integer interval_strides;
    public static SDIndex interval(Integer begin, Integer strides, Integer end){
        if(strides == 0){
            throw new ND4JIllegalArgumentException("Invalid index : strides can not be 0.");
        }
        SDIndex sdIndex = new SDIndex();
        sdIndex.indexType = IndexType.INTERVAL;
        sdIndex.interval_begin = begin;
        sdIndex.interval_end = end;
        sdIndex.interval_strides = strides;
        return sdIndex;
    }
    
}
