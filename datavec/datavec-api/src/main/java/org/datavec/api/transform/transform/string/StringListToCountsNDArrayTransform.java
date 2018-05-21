package org.datavec.api.transform.transform.string;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.apache.commons.io.FileUtils;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.NDArrayMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseTransform;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Converts String column into a bag-of-words (BOW) represented as an NDArray of "counts."<br>
 * Note that the original column is removed in the process
 *
 * @author dave@skymind.io
 */
@JsonIgnoreProperties({"inputSchema", "map", "columnIdx"})
@EqualsAndHashCode(callSuper = false, exclude = {"columnIdx"})
@Data
public class StringListToCountsNDArrayTransform extends BaseTransform {
    protected final String columnName;
    protected final String newColumnName;
    protected final List<String> vocabulary;
    protected final String delimiter;
    protected final boolean binary;
    protected final boolean ignoreUnknown;

    protected final Map<String, Integer> map;

    protected int columnIdx = -1;

    /**
     * @param columnName     The name of the column to convert
     * @param vocabulary     The possible tokens that may be present.
     * @param delimiter      The delimiter for the Strings to convert
     * @param ignoreUnknown  Whether to ignore unknown tokens
     */
    public StringListToCountsNDArrayTransform(String columnName, List<String> vocabulary, String delimiter,
                    boolean binary, boolean ignoreUnknown) {
        this(columnName, columnName + "[BOW]", vocabulary, delimiter, binary, ignoreUnknown);
    }

    /**
     * @param columnName     The name of the column to convert
     * @param vocabulary     The possible tokens that may be present.
     * @param delimiter      The delimiter for the Strings to convert
     * @param ignoreUnknown  Whether to ignore unknown tokens
     */
    public StringListToCountsNDArrayTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("newColumnName") String newColumnName,
                    @JsonProperty("vocabulary") List<String> vocabulary, @JsonProperty("delimiter") String delimiter,
                    @JsonProperty("binary") boolean binary, @JsonProperty("ignoreUnknown") boolean ignoreUnknown) {
        this.columnName = columnName;
        this.newColumnName = newColumnName;
        this.vocabulary = vocabulary;
        this.delimiter = delimiter;
        this.binary = binary;
        this.ignoreUnknown = ignoreUnknown;

        map = new HashMap<>();
        for (int i = 0; i < vocabulary.size(); i++) {
            map.put(vocabulary.get(i), i);
        }
    }

    public static List<String> readVocabFromFile(String path) throws IOException {
        return FileUtils.readLines(new File(path), "utf-8");
    }

    @Override
    public Schema transform(Schema inputSchema) {

        int colIdx = inputSchema.getIndexOfColumn(columnName);

        List<ColumnMetaData> oldMeta = inputSchema.getColumnMetaData();
        List<ColumnMetaData> newMeta = new ArrayList<>();
        List<String> oldNames = inputSchema.getColumnNames();

        Iterator<ColumnMetaData> typesIter = oldMeta.iterator();
        Iterator<String> namesIter = oldNames.iterator();

        int i = 0;
        while (typesIter.hasNext()) {
            ColumnMetaData t = typesIter.next();
            String name = namesIter.next();
            if (i++ == colIdx) {
                //Replace String column with a set of binary/integer columns
                if (t.getColumnType() != ColumnType.String)
                    throw new IllegalStateException("Cannot convert non-string type");

                ColumnMetaData meta = new NDArrayMetaData(newColumnName, new long[] {vocabulary.size()});
                newMeta.add(meta);
            } else {
                newMeta.add(t);
            }
        }

        return inputSchema.newSchema(newMeta);

    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        this.inputSchema = inputSchema;
        this.columnIdx = inputSchema.getIndexOfColumn(columnName);
    }

    @Override
    public String toString() {
        return "StringListToCountsTransform(columnName=" + columnName + ",vocabularySize=" + vocabulary.size()
                        + ",delimiter=\"" + delimiter + "\")";
    }

    protected Collection<Integer> getIndices(String text) {
        Collection<Integer> indices;
        if (binary)
            indices = new HashSet<>();
        else
            indices = new ArrayList<>();
        if (text != null && !text.isEmpty()) {
            String[] split = text.split(delimiter);
            for (String s : split) {
                Integer idx = map.get(s);
                if (idx == null && !ignoreUnknown)
                    throw new IllegalStateException("Encountered unknown String: \"" + s + "\"");
                else if (idx != null)
                    indices.add(idx);
            }
        }
        return indices;
    }

    protected INDArray makeBOWNDArray(Collection<Integer> indices) {
        INDArray counts = Nd4j.zeros(vocabulary.size());
        for (Integer idx : indices)
            counts.putScalar(idx, counts.getDouble(idx) + 1);
        Nd4j.getExecutioner().commit();
        return counts;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if (writables.size() != inputSchema.numColumns()) {
            throw new IllegalStateException("Cannot execute transform: input writables list length (" + writables.size()
                            + ") does not " + "match expected number of elements (schema: " + inputSchema.numColumns()
                            + "). Transform = " + toString());
        }
        int n = writables.size();
        List<Writable> out = new ArrayList<>(n);

        int i = 0;
        for (Writable w : writables) {
            if (i++ == columnIdx) {
                String text = w.toString();
                Collection<Integer> indices = getIndices(text);
                INDArray counts = makeBOWNDArray(indices);
                out.add(new NDArrayWritable(counts));
            } else {
                //No change to this column
                out.add(w);
            }
        }

        return out;
    }

    /**
     * Transform an object
     * in to another object
     *
     * @param input the record to transform
     * @return the transformed writable
     */
    @Override
    public Object map(Object input) {
        return null;
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        return null;
    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        throw new UnsupportedOperationException("New column names is always more than 1 in length");
    }

    /**
     * The output column names
     * This will often be the same as the input
     *
     * @return the output column names
     */
    @Override
    public String[] outputColumnNames() {
        return vocabulary.toArray(new String[vocabulary.size()]);
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return new String[] {columnName()};
    }

    /**
     * Returns a singular column name
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String columnName() {
        return columnName();
    }
}
