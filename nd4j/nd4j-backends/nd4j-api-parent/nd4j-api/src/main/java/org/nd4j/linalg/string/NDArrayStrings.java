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

package org.nd4j.linalg.string;

import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.StringUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;

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
@Getter
@Setter
public class NDArrayStrings {

    public static final String EMPTY_ARRAY_STR = "[]";

    private static final String[] OPEN_BRACKETS =  new String[]{"", "[", "[[", "[[[", "[[[[", "[[[[[", "[[[[[[", "[[[[[[[", "[[[[[[[["};
    private static final String[] CLOSE_BRACKETS = new String[]{"", "]", "]]", "]]]", "]]]]", "]]]]]", "]]]]]]", "]]]]]]]", "]]]]]]]]"};

    /**
     * The default number of elements for printing INDArrays (via NDArrayStrings or INDArray.toString)
     */
    public static final long DEFAULT_MAX_PRINT_ELEMENTS = 1000;
    /**
     * The maximum number of elements to print by default for INDArray.toString()
     * Default value is 1000 - given by {@link #DEFAULT_MAX_PRINT_ELEMENTS}
     */
    @Setter @Getter
    private static long maxPrintElements = DEFAULT_MAX_PRINT_ELEMENTS;

    private long localMaxPrintElements = maxPrintElements;
    private String colSep = ",";
    private String newLineSep = ",";
    private int padding = 7;
    private int precision = 4;
    private double minToPrintWithoutSwitching;
    private double maxToPrintWithoutSwitching;
    private String scientificFormat = "";
    private DecimalFormat decimalFormat;
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

    public NDArrayStrings(long maxElements, int precision) {
        this(",", precision);
        this.localMaxPrintElements = maxElements;
    }

    public NDArrayStrings(long maxElements) {
        this();
        this.localMaxPrintElements = maxElements;
    }

    public NDArrayStrings(long maxElements, boolean forceSummarize, int precision) {
        this(",", precision);
        if(forceSummarize)
            localMaxPrintElements = 0;
        else
            localMaxPrintElements = maxElements;
    }

    public NDArrayStrings(boolean forceSummarize, int precision) {
        this(",", precision);
        if(forceSummarize)
            localMaxPrintElements = 0;
    }



    public NDArrayStrings(boolean forceSummarize) {
        this(",", 4);
        if(forceSummarize)
            localMaxPrintElements = 0;
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
        StringBuilder decFormatNum = new StringBuilder("0.");

        int prec = Math.abs(precision);
        this.precision = prec;
        boolean useHash = precision < 0;

        while (prec > 0) {
            decFormatNum.append(useHash ? "#" : "0");
            prec -= 1;
        }
        this.decimalFormat = localeIndifferentDecimalFormat(decFormatNum.toString());
    }

    /**
     * Specify a col separator and a decimal format string
     * @param colSep
     * @param decFormat
     */
    public NDArrayStrings(String colSep, String decFormat) {
        this.colSep = colSep;
        this.decimalFormat = localeIndifferentDecimalFormat(decFormat);
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
        if(arr.isEmpty())
            return EMPTY_ARRAY_STR;
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
        return format(arr, 0, summarize && arr.length() > localMaxPrintElements);
    }

    private String format(INDArray arr, int offset, boolean summarize) {
        int rank = arr.rank();
        if (arr.isScalar() || arr.length() == 1) {
            int fRank = Math.min(rank, OPEN_BRACKETS.length-1);
            if (arr.isR()) {
                double arrElement = arr.getDouble(0);
                if (!dontOverrideFormat && ((Math.abs(arrElement) < this.minToPrintWithoutSwitching && arrElement != 0) || (Math.abs(arrElement) >= this.maxToPrintWithoutSwitching))) {
                    //switch to scientific notation
                    String asString = localeIndifferentDecimalFormat(scientificFormat).format(arrElement);
                    //from E to small e
                    asString = asString.replace('E', 'e');
                    return OPEN_BRACKETS[fRank] + asString + CLOSE_BRACKETS[fRank];
                } else {
                    if (arr.getDouble(0) == 0) return OPEN_BRACKETS[fRank] + "0" + CLOSE_BRACKETS[fRank];
                    return OPEN_BRACKETS[fRank] + decimalFormat.format(arr.getDouble(0)) + CLOSE_BRACKETS[fRank];
                }
            } else if (arr.isZ()) {
                long arrElement = arr.getLong(0);
                return OPEN_BRACKETS[fRank] + arrElement + CLOSE_BRACKETS[fRank];
            } else if (arr.isB()) {
                long arrElement = arr.getLong(0);
                return OPEN_BRACKETS[fRank] + (arrElement == 0 ? "false" : "true") + CLOSE_BRACKETS[fRank];
            } else if (arr.dataType() == DataType.UTF8){
                String s = arr.getString(0);
                return OPEN_BRACKETS[fRank] + "\"" + s.replaceAll("\n","\\n") + "\"" + CLOSE_BRACKETS[fRank];
            } else
                throw new ND4JIllegalStateException();
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
                    }
//                    else if(arr.rank() <= 1 || arr.length() == 1) {
//                        sb.append(format(Nd4j.scalar(arr.getDouble(0)),offset,summarize));
//                    }
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
            if (summarize && i > 2 && i < l - 3) {
                sb.append("  ...");
                // immediately jump to the last elements so we only print ellipsis once
                i = Math.max(i, (int) l - 4);
            } else {
                if (arr.isR()) {
                    double arrElement = arr.getDouble(i);
                    if (!dontOverrideFormat && ((Math.abs(arrElement) < this.minToPrintWithoutSwitching && arrElement != 0) || (Math.abs(arrElement) >= this.maxToPrintWithoutSwitching))) {
                        //switch to scientific notation
                        String asString = localeIndifferentDecimalFormat(scientificFormat).format(arrElement);
                        //from E to small e
                        asString = asString.replace('E', 'e');
                        sb.append(String.format("%1$" + padding + "s", asString));
                    } else {
                        if (arrElement == 0) {
                            sb.append(String.format("%1$" + padding + "s", 0));
                        } else {
                            sb.append(String.format("%1$" + padding + "s", decimalFormat.format(arrElement)));
                        }
                    }
                } else if (arr.isZ()) {
                    long arrElement = arr.getLong(i);
                    sb.append(String.format("%1$" + padding + "s", arrElement));
                } else if (arr.isB()) {
                    long arrElement = arr.getLong(i);
                    sb.append(String.format("%1$" + padding + "s", arrElement == 0 ? "false" : "true"));
                } else if(arr.dataType() == DataType.UTF8){
                    String s = arr.getString(i);
                    s = "\"" + s.replaceAll("\n", "\\n") + "\"";
                    sb.append(s);
                }
            }
            if (i < l - 1) {
                if (!summarize || i <= 2 || i >= l - 3 || (summarize && l == 6)) {
                    sb.append(colSep);
                }
            }
        }
        sb.append("]");
        return sb.toString();
    }


    private DecimalFormat localeIndifferentDecimalFormat(String pattern){
        return new DecimalFormat(pattern, DecimalFormatSymbols.getInstance(Locale.US));
    }
}
