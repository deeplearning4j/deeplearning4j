package org.nd4j.linalg.util;

import org.nd4j.base.PreconditionsFormat;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.List;

/**
 * Preconditions format: Defines a set of tags for use with {@link org.nd4j.base.Preconditions} class.<br>
 * %ndRank: rank of INDArray<br>
 * %ndShape: shape of INDArray<br>
 * %ndStride: stride of INDArray<br>
 * %ndLength: length of INDArray<br>
 * %ndSInfo: shape info of INDArray<br>
 * %nd10: First 10 values of INDArray (or all values if length <= 10<br>
 *
 * @author Alex Black
 */
public class NDArrayPreconditionsFormat implements PreconditionsFormat {

    private static final List<String> TAGS = Arrays.asList(
            "%ndRank", "%ndShape", "%ndStride", "%ndLength", "%ndSInfo", "%nd10");

    @Override
    public List<String> formatTags() {
        return TAGS;
    }

    @Override
    public String format(String tag, Object arg) {
        if(arg == null)
            return "null";
        INDArray arr = (INDArray)arg;
        switch (tag){
            case "%ndRank":
                return String.valueOf(arr.rank());
            case "%ndShape":
                return Arrays.toString(arr.shape());
            case "%ndStride":
                return Arrays.toString(arr.stride());
            case "%ndLength":
                return String.valueOf(arr.length());
            case "%ndSInfo":
                return arr.shapeInfoToString().replaceAll("\n","");
            case "%nd10":
                if(arr.isScalar() || arr.isEmpty()){
                    return arr.toString();
                }
                INDArray sub = arr.reshape(arr.length()).get(NDArrayIndex.interval(0, Math.min(arr.length(), 10)));
                return sub.toString();
            default:
                //Should never happen
                throw new IllegalStateException("Unknown format tag: " + tag);
        }
    }
}
