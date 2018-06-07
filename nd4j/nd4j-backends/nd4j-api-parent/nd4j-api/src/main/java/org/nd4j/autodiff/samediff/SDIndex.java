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
    private Long intervalBegin;
    private Long intervalEnd;
    private Long intervalStrides;




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
    
}
