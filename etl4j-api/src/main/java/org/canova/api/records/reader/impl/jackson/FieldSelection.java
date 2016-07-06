package org.canova.api.records.reader.impl.jackson;

import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;

import java.util.ArrayList;
import java.util.List;

/**
 * FieldSelection is used in conjunction with the {@link JacksonRecordReader} (and the subclasses).
 *
 * The are a few motivations here:<br>
 * - Formats such as XML, JSON and YAML can contain arbitrarily nested components, and we need to flatten them somehow<br>
 * - These formats can vary in terms of order (for example, JSON is unordered), so we need to define the exact order of outputs for the record reader<br>
 * - In any given JSON/XML/YAML file, there might not be a particular value present (but: we still want it to be represented in the output)<br>
 * - In any given JSON/XML/YAML file, we might want to totally ignore certain fields<br>
 *
 * @author Alex Black
 */
public class FieldSelection {

    public static final Writable DEFAULT_MISSING_VALUE = new Text("");

    private List<String[]> fieldPaths;
    private List<Writable> valueIfMissing;

    private FieldSelection(Builder builder){
        this.fieldPaths = builder.fieldPaths;
        this.valueIfMissing = builder.valueIfMissing;
    }

    public List<String[]> getFieldPaths(){
        return fieldPaths;
    }

    public List<Writable> getValueIfMissing(){
        return valueIfMissing;
    }

    public int getNumFields(){
        return fieldPaths.size();
    }

    public static class Builder {

        private List<String[]> fieldPaths = new ArrayList<>();
        private List<Writable> valueIfMissing = new ArrayList<>();


        /**
         *
         * @param fieldPath    Path to the field. For example, {"a", "b", "c"} would be a field named c, in an object b,
         *                     where b is in an object a
         */
        public Builder addField(String... fieldPath){
            return addField(DEFAULT_MISSING_VALUE, fieldPath);
        }

        public Builder addField(Writable valueIfMissing, String... fieldPath){
            fieldPaths.add(fieldPath);
            this.valueIfMissing.add(valueIfMissing);
            return this;
        }

        public FieldSelection build(){
            return new FieldSelection(this);
        }
    }

}
