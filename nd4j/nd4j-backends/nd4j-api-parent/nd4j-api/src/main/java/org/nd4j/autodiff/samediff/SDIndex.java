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
