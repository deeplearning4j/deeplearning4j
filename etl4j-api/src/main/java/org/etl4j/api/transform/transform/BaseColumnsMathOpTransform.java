package io.skymind.echidna.api.transform;

import org.canova.api.writable.Writable;
import io.skymind.echidna.api.MathOp;
import io.skymind.echidna.api.Transform;
import io.skymind.echidna.api.metadata.ColumnMetaData;
import io.skymind.echidna.api.schema.Schema;
import io.skymind.echidna.api.transform.integer.IntegerMathOpTransform;
import io.skymind.echidna.api.transform.longtransform.LongMathOpTransform;
import io.skymind.echidna.api.transform.real.DoubleMathOpTransform;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Base class for multiple column math operations. For example myNewColumn = column1 + column2 + column3.<br>
 * Note: this transform adds a (derived) column (with the specified name) to the end<br>
 * <p/>
 * Note: Certain operations have restrictions on the number of columns used as input<br>
 * Add: 2 or more (a+b+c+...)<br>
 * Subtract: exactly 2 (a-b)<br>
 * Multiply: 2 or more (a*b*c*...)<br>
 * Divide: exactly 2 (a/b)<br>
 * Modulus: exactly 2 (a%b)<br>
 * Other math ops (ReverseSubtract, ReverseDivide, ScalarMin, ScalarMax) are not allowed here.
 * <p/>
 * <p/>
 * <p/>
 * <br><br>
 * <b>See</b>: {@link IntegerMathOpTransform}, {@link DoubleMathOpTransform}, {@link LongMathOpTransform} for operations
 * with a scalar + single column, instea
 */
public abstract class BaseColumnsMathOpTransform implements Transform {

    protected final String newColumnName;
    protected final MathOp mathOp;
    protected final String[] columns;
    private int[] columnIdxs;
    private Schema inputSchema;

    public BaseColumnsMathOpTransform(String newColumnName, MathOp mathOp, String... columns) {
        if (columns == null || columns.length == 0) throw new IllegalArgumentException("Invalid input: cannot have null/0 columns");
        this.newColumnName = newColumnName;
        this.mathOp = mathOp;
        this.columns = columns;

        switch (mathOp) {
            case Add:
                if (columns.length < 2)
                    throw new IllegalArgumentException("Need 2 or more columns for Add op. Got: " + Arrays.toString(columns));
                break;
            case Subtract:
                if (columns.length != 2)
                    throw new IllegalArgumentException("Need exactly 2 columns for Subtract op. Got: " + Arrays.toString(columns));
                break;
            case Multiply:
                if (columns.length < 2)
                    throw new IllegalArgumentException("Need 2 or more columns for Multiply op. Got: " + Arrays.toString(columns));
                break;
            case Divide:
                if (columns.length != 2)
                    throw new IllegalArgumentException("Need exactly 2 columns for Divide op. Got: " + Arrays.toString(columns));
                break;
            case Modulus:
                if (columns.length != 2)
                    throw new IllegalArgumentException("Need exactly 2 columns for Modulus op. Got: " + Arrays.toString(columns));
                break;
            case ReverseSubtract:
            case ReverseDivide:
            case ScalarMin:
            case ScalarMax:
                throw new IllegalArgumentException("Invalid MathOp: cannot use " + mathOp + " with ...ColumnsMathOpTransform");
            default:
                throw new RuntimeException("Unknown MathOp: " + mathOp);
        }
    }

    @Override
    public Schema transform(Schema inputSchema) {
        for (String name : columns) {
            if (!inputSchema.hasColumn(name))
                throw new IllegalStateException("Input schema does not have column with name \"" + name + "\"");
        }

        List<String> newNames = new ArrayList<>(inputSchema.getColumnNames());
        List<ColumnMetaData> newMeta = new ArrayList<>(inputSchema.getColumnMetaData());

        newNames.add(newColumnName);
        newMeta.add(derivedColumnMetaData());

        return inputSchema.newSchema(newNames, newMeta);
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        columnIdxs = new int[columns.length];
        int i = 0;
        for (String name : columns) {
            if (!inputSchema.hasColumn(name))
                throw new IllegalStateException("Input schema does not have column with name \"" + name + "\"");
            columnIdxs[i++] = inputSchema.getIndexOfColumn(name);
        }

        this.inputSchema = inputSchema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if (inputSchema == null) throw new IllegalStateException("Input schema has not been set");
        List<Writable> out = new ArrayList<>(writables);

        Writable[] temp = new Writable[columns.length];
        for( int i=0; i<columnIdxs.length; i++ ) temp[i] = out.get(columnIdxs[i]);

        out.add(doOp(temp));
        return out;
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        List<List<Writable>> out = new ArrayList<>();
        for (List<Writable> step : sequence) {
            out.add(map(step));
        }
        return out;
    }

    protected abstract ColumnMetaData derivedColumnMetaData();

    protected abstract Writable doOp(Writable... input);
}
