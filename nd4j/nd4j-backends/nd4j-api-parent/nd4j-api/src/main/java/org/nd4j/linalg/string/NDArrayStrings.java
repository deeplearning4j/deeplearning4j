package org.nd4j.linalg.string;

import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.text.DecimalFormat;

/**
 *  String representation of an ndarray.
 *
 * Printing will default to scientific notation on a per element basis
 *      - when absolute value is greater than or equal to 10000
 *      - when absolute value is less than or equal to 0.0001
 *
 *  If the number of elements in the array is greater than 1000 only the first and last three elements in a dimension are included
 *
 * @author Adam Gibson
 * @author Susan Eraly
 */
public class NDArrayStrings {

    private String colSep = ",";
    private String newLineSep = ",";
    private int padding = 7;
    private int precision = 4;
    private double minToPrintWithoutSwitching;
    private double maxToPrintWithoutSwitching;
    private String scientificFormat = "";
    private DecimalFormat decimalFormat = new DecimalFormat("##0.####");
    private boolean dontOverrideFormat = false;

    public NDArrayStrings() {
        this(",", 4);
    }

    public NDArrayStrings(String colSep) {
        this(colSep, 4);
    }

    /**
     * Specify the number of digits after the decimal point to include
     * @param precision
     */
    public NDArrayStrings(int precision) {
        this(",", precision);
    }


    /**
     * Specify a delimiter for elements in columns for 2d arrays (or in the rank-1th dimension in higher order arrays)
     * Separator in elements in remaining dimensions defaults to ",\n"
     *
     * @param colSep    field separating columns;
     * @param precision digits after decimal point
     */
    public NDArrayStrings(String colSep, int precision) {
        this.colSep = colSep;
        if (!colSep.replaceAll("\\s", "").equals(",")) this.newLineSep = "";
        this.precision = precision;
        String decFormatNum = "0.";
        while (precision > 0) {
            decFormatNum += "0";
            precision -= 1;
        }
        this.decimalFormat = new DecimalFormat(decFormatNum);
    }

    /**
     * Specify a col separator and a decimal format string
     * @param colSep
     * @param decFormat
     */
    public NDArrayStrings(String colSep, String decFormat) {
        this.colSep = colSep;
        this.decimalFormat = new DecimalFormat(decFormat);
        if (decFormat.toUpperCase().contains("E")) {
            this.padding = decFormat.length() + 3;
        } else {
            this.padding = decFormat.length() + 1;
        }
        this.dontOverrideFormat = true;
    }

    /**
     *
     * @param arr
     * @return String representation of the array adhering to options provided in the constructor
     */
    public String format(INDArray arr) {
        return format(arr, true);
    }

    /**
     * Format the given ndarray as a string
     *
     * @param arr       the array to format
     * @param summarize If true and the number of elements in the array is greater than > 1000 only the first three and last elements in any dimension will print
     * @return the formatted array
     */
    public String format(INDArray arr, boolean summarize) {
        this.scientificFormat = "0.";
        int addPrecision = this.precision;
        while (addPrecision > 0) {
            this.scientificFormat += "#";
            addPrecision -= 1;
        }
        this.scientificFormat = this.scientificFormat + "E0";
        if (this.scientificFormat.length() + 2  > this.padding) this.padding = this.scientificFormat.length() + 2;
        this.maxToPrintWithoutSwitching = Math.pow(10,this.precision);
        this.minToPrintWithoutSwitching = 1.0/(this.maxToPrintWithoutSwitching);
        if (summarize && arr.length() > 1000) return format(arr, 0, true);
        return format(arr, 0, false);
    }

    private String format(INDArray arr, int offset, boolean summarize) {
        int rank = arr.rank();
        if (arr.isScalar()) {
            //true scalar i.e shape = [] not legacy which is [1,1]
            if (arr instanceof IComplexNDArray) {
                return ((IComplexNDArray) arr).getComplex(0).toString();
            }
            /////
            double arrElement = arr.getDouble(0);
            if (!dontOverrideFormat && ((Math.abs(arrElement) < this.minToPrintWithoutSwitching && arrElement!= 0) || (Math.abs(arrElement) >= this.maxToPrintWithoutSwitching))) {
                //switch to scientific notation
                String asString = new DecimalFormat(scientificFormat).format(arrElement);
                //from E to small e
                asString = asString.replace('E','e');
                return asString;
            }
            else {
                if (arr.getDouble(0) == 0) return "0";
                return decimalFormat.format(arr.getDouble(0));
            }
        } else if (rank == 1) {
            //true vector
            return vectorToString(arr, summarize);
        } else if (arr.isRowVector()) {
            //a slice from a higher dim array
            if (offset == 0) {
                StringBuilder sb = new StringBuilder();
                sb.append("[");
                sb.append(vectorToString(arr, summarize));
                sb.append("]");
                return sb.toString();
            }
            return vectorToString(arr, summarize);
        } else {
            offset++;
            StringBuilder sb = new StringBuilder();
            sb.append("[");
            long nSlices = arr.slices();
            for (int i = 0; i < nSlices; i++) {
                if (summarize && i > 2 && i < nSlices - 3) {
                    sb.append(" ...");
                    sb.append(newLineSep).append(" \n");
                    sb.append(StringUtils.repeat("\n", rank - 2));
                    sb.append(StringUtils.repeat(" ", offset));
                    // immediately jump to the last slices so we only print ellipsis once
                    i = Math.max(i, (int) nSlices - 4);
                } else {
                    if (arr.rank() == 3 && arr.slice(i).isRowVector()) sb.append("[");
                    //hack fix for slice issue with 'f' order
                    if (arr.ordering() == 'f' && arr.rank() > 2 && arr.size(arr.rank() - 1) == 1) {
                        sb.append(format(arr.dup('c').slice(i), offset, summarize));
                    } else if(arr.rank() <= 1 || arr.length() == 1) {
                        sb.append(format(Nd4j.scalar(arr.getDouble(0)),offset,summarize));
                    }
                    else {
                        sb.append(format(arr.slice(i), offset, summarize));
                    }
                    if (i != nSlices - 1) {
                        if (arr.rank() == 3 && arr.slice(i).isRowVector()) sb.append("]");
                        sb.append(newLineSep).append(" \n");
                        sb.append(StringUtils.repeat("\n", rank - 2));
                        sb.append(StringUtils.repeat(" ", offset));
                    } else {
                        if (arr.rank() == 3 && arr.slice(i).isRowVector()) sb.append("]");
                    }
                }
            }
            sb.append("]");
            return sb.toString();
        }
    }

    private String vectorToString(INDArray arr, boolean summarize) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        long l = arr.length();
        for (int i = 0; i <l; i++) {
            if (arr instanceof IComplexNDArray) {
                sb.append(((IComplexNDArray) arr).getComplex(i).toString());
            } else {
                if (summarize && i > 2 && i < l - 3) {
                    sb.append("  ...");
                    // immediately jump to the last elements so we only print ellipsis once
                    i = Math.max(i, (int) l - 4);
                } else {
                    double arrElement = arr.getDouble(i);
                    if (!dontOverrideFormat && ((Math.abs(arrElement) < this.minToPrintWithoutSwitching && arrElement!= 0) || (Math.abs(arrElement) >= this.maxToPrintWithoutSwitching))) {
                        //switch to scientific notation
                        String asString = new DecimalFormat(scientificFormat).format(arrElement);
                        //from E to small e
                        asString = asString.replace('E','e');
                        sb.append(String.format("%1$" + padding + "s", asString));
                    }
                    else {
                        if (arrElement == 0) {
                            sb.append(String.format("%1$" + padding + "s", 0));
                        }
                        else {
                            sb.append(String.format("%1$" + padding + "s", decimalFormat.format(arrElement)));
                        }
                    }
                }
            }
            if (i < l - 1) {
                if (!summarize || i < 2 || i > l - 3 || (summarize && l == 6)) {
                    sb.append(colSep);
                }
            }
        }
        sb.append("]");
        return sb.toString();
    }

}
