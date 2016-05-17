package org.nd4j.linalg.string;

import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.text.DecimalFormat;

/**
 * @author Adam Gibson
 * @author Susan Eraly
 */
public class NDArrayStrings {
    private DecimalFormat decimalFormat = new DecimalFormat("#,###,##0.00");
    //String format = "%10.2f\n";
    private String sep = ",";
    private int padding = 0;


    public NDArrayStrings(String sep) {
        this.sep = sep;
    }

    public NDArrayStrings() {
        this(", ");
    }


    /**
     * Format the given ndarray as a string
     * @param arr the array to format
     * @return the formatted array
     */
    public String format(INDArray arr) {
        String padding = decimalFormat.format(arr.maxNumber());
        this.padding = padding.length();
        return format(arr,arr.rank());
    }
    private String format(INDArray arr,int rank) {
        return format(arr,arr.rank(),0);
    }

    private String format(INDArray arr,int rank, int offset) {
        StringBuffer sb = new StringBuffer();
        if(arr.isScalar()) {
            if(arr instanceof IComplexNDArray)
                return ((IComplexNDArray) arr).getComplex(0).toString();
            return decimalFormat.format(arr.getDouble(0));
        }
        else if(rank <= 0)
            return "";

        else if(arr.isVector()) {
            sb.append("[");
            for(int i = 0; i < arr.length(); i++) {
                if(arr instanceof IComplexNDArray)
                    sb.append(((IComplexNDArray) arr).getComplex(i).toString());
                else
                    sb.append(String.format("%1$"+padding+"s",decimalFormat.format(arr.getDouble(i))));
                if(i < arr.length() - 1)
                    sb.append(sep);
            }
            sb.append("]");
            return sb.toString();
        }

        else {
            offset++;
            sb.append("[");
            for(int i = 0; i < arr.slices(); i++) {
                sb.append(format(arr.slice(i),rank - 1,offset));
                if (i != arr.slices() - 1) {
                    sb.append(",\n");
                    sb.append(StringUtils.repeat("\n",rank-2));
                    sb.append(StringUtils.repeat(" ",offset));
                }
            }
            sb.append("]");
            return sb.toString();
        }
    }

}
