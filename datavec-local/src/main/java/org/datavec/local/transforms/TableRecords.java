package org.datavec.local.transforms;

import org.datavec.api.transform.*;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.condition.column.*;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.rank.CalculateSortedRank;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import tech.tablesaw.aggregate.AggregateFunction;
import tech.tablesaw.aggregate.AggregateFunctions;
import tech.tablesaw.api.*;
import tech.tablesaw.columns.Column;
import tech.tablesaw.columns.ColumnReference;
import tech.tablesaw.filtering.*;
import tech.tablesaw.filtering.times.IsAfter;
import tech.tablesaw.filtering.times.IsBefore;
import tech.tablesaw.store.ColumnMetadata;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.time.LocalTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Util class for interop between
 * normal datavec records and
 * the dataframe
 *
 * @author Adam Gibson
 */
public class TableRecords {


    /**
     * Execute the specified TransformProcess with the given input data<br>
     * Note: this method can only be used if the TransformProcess returns non-sequence data. For TransformProcesses
     * that return a sequence, use {@link #executeToSequence(JavaRDD, TransformProcess)}
     *
     * @param inputWritables   Input data to process
     * @param transformProcess TransformProcess to executeNonTimeSeries
     * @return Processed data
     */
    public static List<List<Writable>> executeNonTimeSeries(List<List<Writable>> inputWritables,
                    TransformProcess transformProcess) {
        Table table = fromRecordsAndSchema(inputWritables, transformProcess.getInitialSchema());
        Table ret = transformNonTimeSeries(table, transformProcess);
        return fromTable(ret);
    }

    /**
     * Apply a transformNonTimeSeries process
     * to the given table
     * @param table the table to apply this to
     * @param transformProcess the transformNonTimeSeries
     *                         process to apply
     * @return the transformed table
     */
    public static Table transformNonTimeSeries(Table table, TransformProcess transformProcess) {
        List<DataAction> dataActions = transformProcess.getActionList();
        Table ret = table.fullCopy();
        for (DataAction dataAction : dataActions) {
            if (dataAction.getTransform() != null) {
                ret = transformTable(table, dataAction.getTransform());
            } else if (dataAction.getFilter() != null) {
                ret = filterTable(ret, dataAction.getFilter());
            } else if (dataAction.getCalculateSortedRank() != null) {
                ret = sortedRank(ret, dataAction.getCalculateSortedRank());
            } else if (dataAction.getConvertFromSequence() != null) {
                throw new UnsupportedOperationException(
                                "Tables don't have a time series sequence, please use the List<Writable> version of this function for input");

            } else if (dataAction.getReducer() != null) {
                ret = reduce(ret, (Reducer) dataAction.getReducer());
            } else if (dataAction.getSequenceSplit() != null) {
                throw new UnsupportedOperationException(
                                "Tables don't have a time series sequence, please use the List<Writable> version of this function for input");

            }
        }

        return ret;
    }


    private static AggregateFunction getFunction(ReduceOp reduceOp) {
        switch (reduceOp) {
            case Stdev:
                return AggregateFunctions.stdDev;
            case Sum:
                return AggregateFunctions.sum;
            case Max:
                return AggregateFunctions.max;
            case Mean:
                return AggregateFunctions.mean;
            case Min:
                return AggregateFunctions.min;
            case Count:
                return AggregateFunctions.count;
            case CountUnique:
                throw new IllegalArgumentException("Illegal operation " + reduceOp);
            case Range:
                return AggregateFunctions.range;
            case TakeFirst:
                return AggregateFunctions.first;
            case TakeLast:
                return AggregateFunctions.last;
            case Prod:
                return AggregateFunctions.product;
            default:
                throw new IllegalStateException("Illegal operation for reduce");
        }
    }

    /**
     * Run a reduce operation on the given table
     * @param reduce the reduce operation to run
     * @param reducer the reducer to run
     * @return
     */
    public static Table reduce(Table reduce, Reducer reducer) {

        if (reducer.getConditionalReductions() != null) {
            for (Map.Entry<String, Reducer.ConditionalReduction> pair : reducer.getConditionalReductions().entrySet()) {
                Condition conditionToFilter = pair.getValue().getCondition();
                String inputColumnName = pair.getKey();
                Schema output = reducer.transform(reducer.getInputSchema());
                tech.tablesaw.filtering.Filter filter =
                                mapFilterFromCondition(conditionToFilter, inputColumnName, output);
                reduce = runReduce(reduce.selectWhere(filter), pair.getValue().getReductions().get(0), inputColumnName);
            }
        } else {
            for (Map.Entry<String, List<ReduceOp>> pair : reducer.getOpMap().entrySet()) {
                reduce = runReduce(reduce, pair.getValue().get(0), pair.getKey());
            }
        }


        return reduce;
    }


    private static Table runReduce(Table reduce, ReduceOp reduceOp, String columnNameToReduce) {
        switch (reduceOp) {
            case Count:
                return reduce.countBy(reduce.categoryColumn(columnNameToReduce));
            case CountUnique:
                throw new IllegalArgumentException("Illegal operation ");
            default:
                AggregateFunction reduceFunction = getFunction(reduceOp);
                String outputName = columnNameToReduce = "reduced_" + columnNameToReduce;
                switch (reduce.column(columnNameToReduce).type()) {
                    case FLOAT:
                        double val = reduce.agg(columnNameToReduce, reduceFunction);
                        FloatColumn floatColumn
                                = new FloatColumn(columnNameToReduce, new float[] {(float) val});
                        return Table.create(outputName, floatColumn);
                    case LONG_INT:
                        long longVal = (long) reduce.agg(columnNameToReduce, reduceFunction);
                        LongColumn longColumn
                                = new LongColumn(columnNameToReduce, new long[] {longVal});
                        return Table.create(outputName, longColumn);
                    case INTEGER:
                        int intVal = (int) reduce.agg(columnNameToReduce, reduceFunction);
                        IntColumn intColumn
                                = new IntColumn(columnNameToReduce, new int[] {intVal});
                        return Table.create(outputName, intColumn);
                    case DOUBLE:
                        double doubleVal = reduce.agg(columnNameToReduce, reduceFunction);
                        DoubleColumn doubleColumn
                                = new DoubleColumn(columnNameToReduce, new double[] {doubleVal});
                        return Table.create(outputName, doubleColumn);
                    default:
                        throw new IllegalStateException("Illegal column type  " + reduceOp);

                }

        }

    }

    /**
     *
     * @param toRank
     * @param rank
     * @return
     */
    public static Table sortedRank(Table toRank, CalculateSortedRank rank) {
        Table clone = toRank.fullCopy();
        LongColumn longColumn = new LongColumn(rank.outputColumnName(), toRank.rowCount());
        for (int i = 0; i < toRank.rowCount(); i++) {
            longColumn.append(i);
        }

        clone.addColumn(longColumn);


        if (rank.isAscending()) {
            Table sorted = clone.sortAscendingOn(rank.columnNames());
            Table newTable = Table.create("sorted", sorted.column(rank.outputColumnName()));
            return newTable;
        } else {
            Table sorted = clone.sortDescendingOn(rank.columnNames());
            Table newTable = Table.create("sorted", sorted.column(rank.outputColumnName()));
            return newTable;
        }
    }


    /**
     * Map a condition
     * based on an input column
     * and an output schema
     * to a data frame filter
     * @param condition the condition to map
     * @param columnName the input column name to map
     * @param output the output schema to infer the type
     * @return the appropriate dataframe filter
     */
    public static tech.tablesaw.filtering.Filter mapFilterFromCondition(Condition condition, String columnName,
                    Schema output) {
        //map to proper column condition for the filter to apply
        if (condition instanceof ColumnCondition) {
            ColumnCondition columnCondition = (ColumnCondition) condition;
            ColumnReference columnReference = new ColumnReference(columnName);
            switch (output.getType(output.getIndexOfColumn(columnCondition.outputColumnName()))) {
                case String:
                    CategoricalColumnCondition categoricalColumnCondition =
                                    (CategoricalColumnCondition) columnCondition;
                    switch (categoricalColumnCondition.getOp()) {
                        case Equal:
                            return new StringEqualTo(columnReference, categoricalColumnCondition.getValue());
                        case NotEqual:
                            return new StringNotEqualTo(columnReference, categoricalColumnCondition.getValue());
                        case InSet:
                            return new StringIsIn(columnReference, categoricalColumnCondition.getSet());
                        case NotInSet:
                            return new StringIsNotIn(columnReference, categoricalColumnCondition.getSet());

                    }
                case Long:
                    LongColumnCondition longColumnCondition = (LongColumnCondition) columnCondition;
                    switch (longColumnCondition.getOp()) {
                        case Equal:
                            return new LongEqualTo(columnReference, longColumnCondition.getValue().longValue());
                        case NotEqual:
                            return new LongNotEqualTo(columnReference, longColumnCondition.getValue().longValue());
                        case GreaterThan:
                            return new LongGreaterThan(columnReference, longColumnCondition.getValue().longValue());
                        case LessOrEqual:
                            return new LongLessThanOrEqualTo(columnReference,
                                            longColumnCondition.getValue().longValue());
                        case GreaterOrEqual:
                            return new LongGreaterThanOrEqualTo(columnReference,
                                            longColumnCondition.getValue().longValue());
                        case LessThan:
                            return new LongLessThan(columnReference, longColumnCondition.getValue().longValue());
                        default:
                            throw new IllegalStateException("Illegal operation ");
                    }
                case Categorical:
                    CategoricalColumnCondition categoricalColumnCondition2 =
                                    (CategoricalColumnCondition) columnCondition;
                    switch (categoricalColumnCondition2.getOp()) {
                        case Equal:
                            return new StringEqualTo(columnReference, categoricalColumnCondition2.getValue());
                        case NotEqual:
                            return new StringNotEqualTo(columnReference, categoricalColumnCondition2.getValue());
                        case InSet:
                            return new StringIsIn(columnReference, categoricalColumnCondition2.getSet());
                        case NotInSet:
                            return new StringIsNotIn(columnReference, categoricalColumnCondition2.getSet());

                    }
                case Float:
                    DoubleColumnCondition floatColumnCondition = (DoubleColumnCondition) columnCondition;
                    switch (floatColumnCondition.getOp()) {
                        case Equal:
                            return new FloatEqualTo(columnReference, floatColumnCondition.getValue().floatValue());
                        case NotEqual:
                            return new FloatNotEqualTo(columnReference, floatColumnCondition.getValue().floatValue());
                        case GreaterThan:
                            return new FloatGreaterThan(columnReference, floatColumnCondition.getValue().floatValue());
                        case LessOrEqual:
                            return new FloatGreaterThanOrEqualTo(columnReference,
                                            floatColumnCondition.getValue().floatValue());
                        case GreaterOrEqual:
                            return new FloatGreaterThanOrEqualTo(columnReference,
                                            floatColumnCondition.getValue().floatValue());
                        case LessThan:
                            return new FloatLessThan(columnReference, floatColumnCondition.getValue().floatValue());
                        default:
                            throw new IllegalStateException("Illegal operation ");
                    }
                case Time:
                    TimeColumnCondition timeColumnCondition = (TimeColumnCondition) columnCondition;
                    switch (timeColumnCondition.getOp()) {
                        case Equal:
                            return new TimeEqualTo(columnReference,
                                            LocalTime.ofNanoOfDay(timeColumnCondition.getValue().longValue()));
                        case NotEqual:
                            return new TimeNotEqualTo(columnReference,
                                            LocalTime.ofNanoOfDay(timeColumnCondition.getValue().longValue()));
                        case GreaterThan:
                            return new IsAfter(columnReference,
                                            LocalTime.ofNanoOfDay(timeColumnCondition.getValue().longValue()));
                        case LessThan:
                            return new IsBefore(columnReference,
                                            LocalTime.ofNanoOfDay(timeColumnCondition.getValue().longValue()));
                        default:
                            throw new IllegalStateException("Illegal operation ");
                    }
                case Boolean:
                    BooleanColumnCondition booleanColumnCondition = (BooleanColumnCondition) columnCondition;
                    return new BooleanIsTrue(columnReference);
                case Double:
                    DoubleColumnCondition doubleColumnCondition = (DoubleColumnCondition) columnCondition;
                    switch (doubleColumnCondition.getOp()) {
                        case Equal:
                            return new DoubleEqualTo(columnReference, doubleColumnCondition.getValue().doubleValue());
                        case NotEqual:
                            return new DoubleNotEqualTo(columnReference,
                                            doubleColumnCondition.getValue().doubleValue());
                        case GreaterThan:
                            return new DoubleGreaterThan(columnReference,
                                            doubleColumnCondition.getValue().doubleValue());
                        case LessOrEqual:
                            return new DoubleLessThanOrEqualTo(columnReference,
                                            doubleColumnCondition.getValue().doubleValue());
                        case GreaterOrEqual:
                            return new DoubleGreaterThanOrEqualTo(columnReference,
                                            doubleColumnCondition.getValue().doubleValue());
                        case LessThan:
                            return new DoubleLessThan(columnReference, doubleColumnCondition.getValue().doubleValue());
                        default:
                            throw new IllegalStateException("Illegal operation ");
                    }
                case Integer:
                    IntegerColumnCondition integerColumnCondition = (IntegerColumnCondition) columnCondition;
                    switch (integerColumnCondition.getOp()) {
                        case Equal:
                            return new IntEqualTo(columnReference, integerColumnCondition.getValue().intValue());
                        case NotEqual:
                            return new IntNotEqualTo(columnReference, integerColumnCondition.getValue().intValue());
                        case GreaterThan:
                            return new IntGreaterThan(columnReference, integerColumnCondition.getValue().intValue());
                        case LessOrEqual:
                            return new IntLessThanOrEqualTo(columnReference,
                                            integerColumnCondition.getValue().intValue());
                        case GreaterOrEqual:
                            return new IntGreaterThanOrEqualTo(columnReference,
                                            integerColumnCondition.getValue().intValue());
                        case LessThan:
                            return new IntLessThan(columnReference, integerColumnCondition.getValue().intValue());
                        default:
                            throw new IllegalStateException("Illegal operation ");
                    }
                default:
                    throw new IllegalArgumentException("Illegal type");
            }

        }
        return null;
    }


    /**
     * Map a given filter
     * to a dataframe filter
     * @param toMap the filter to map
     * @return an apporiate {@link tech.tablesaw.filtering.Filter}
     * mapping from the datavec api
     */
    public static tech.tablesaw.filtering.Filter mapToFilter(Filter toMap) {
        ConditionFilter conditionFilter = (ConditionFilter) toMap;
        Condition condition = conditionFilter.getCondition();
        Schema output = toMap.transform(toMap.getInputSchema());
        return mapFilterFromCondition(condition, toMap.columnName(), output);
    }

    /**
     * Implements a filter
     * @param toFilter
     * @param filter
     * @return
     */
    public static Table filterTable(Table toFilter, Filter filter) {
        Table ret = toFilter;
        List<Integer> indicesToRemove = new ArrayList<>();
        for (String columnName : filter.columnNames()) {
            Column column = toFilter.column(columnName);
            for (int i = 0; i < ret.rowCount(); i++) {
                Object curr = getEntry(column, i);
                if (filter.removeExample(curr)) {
                    indicesToRemove.add(i);
                }
            }
        }

        ret = toFilter.dropRows(indicesToRemove);

        return ret;

    }

    /**
     * Run a transformNonTimeSeries operation on the table
     * @param table the table to run the transformNonTimeSeries operation on
     * @param transform the transformNonTimeSeries to run
     * @return
     */
    public static Table transformTable(Table table, Transform transform) {
        if (!(transform instanceof ColumnOp)) {
            throw new IllegalArgumentException("Transform operation must be of type ColumnOp");
        }

        Schema outputSchema = transform.transform(transform.getInputSchema());
        Table ret = tableFromSchema(outputSchema);
        //copy over data
        for (Column c : ret.columns()) {
            if (table.columnNames().contains(c.name()))
                c.append(table.column(c.name()));
        }



        String[] columnNames = transform.columnNames();
        String[] newColumnNames = transform.outputColumnNames();
        List<Column> inputColumns = table.columns(columnNames);
        List<Column> outputColumns = ret.columns(newColumnNames);
        //a + b -> c: many to 1
        if (columnNames.length > newColumnNames.length) {
            for (int r = 0; r < ret.rowCount(); r++) {
                //set the value in the column for each row
                Object output = transform.map(determineInput(r, inputColumns.toArray(new Column[inputColumns.size()])));
                setEntry(outputColumns.get(0), r, output);
            }

        }
        //a -> a_1,a_2,a_3,...
        else if (columnNames.length < newColumnNames.length) {
            for (int r = 0; r < ret.rowCount(); r++) {
                //set the value in the column for each row
                Object output = transform.map(determineInput(r, inputColumns.toArray(new Column[inputColumns.size()])));
                setEntryList(outputColumns.toArray(new Column[outputColumns.size()]), r, output);
            }

        }

        else {
            //1 to 1 case
            boolean sameTypesForOutput =
                            transform.getInputSchema().sameTypes(transform.transform(transform.getInputSchema()));
            for (String columnName : columnNames) {
                Column column = table.column(columnName);
                Column retColumn = ret.column(columnName);
                if (column instanceof FloatColumn) {
                    FloatColumn floatColumn = (FloatColumn) column;
                    FloatColumn retFloatColumn = (FloatColumn) retColumn;
                    if (sameTypesForOutput)
                        for (int i = 0; i < floatColumn.size(); i++) {
                            retFloatColumn.set(i, (Float) transform.map(floatColumn.get(i)));
                        }
                    else {
                        //remove the column and append new columns on to the end.
                        //map is going to produce more than 1 output it will be easier to add it to the end
                        ret.removeColumn(ret.columnIndex(retColumn));
                        for (int i = 0; i < floatColumn.size(); i++) {
                            //infer types from the column metadata
                            Object output = transform.map(floatColumn.get(i));
                            setEntry(retColumn, i, output);
                        }

                    }

                } else if (column instanceof LongColumn) {
                    LongColumn longColumn = (LongColumn) column;
                    LongColumn retLongColumn = (LongColumn) retColumn;
                    if (sameTypesForOutput)
                        for (int i = 0; i < longColumn.size(); i++) {
                            retLongColumn.set(i, (Long) transform.map(longColumn.get(i)));
                        }
                    else {
                        //remove the column and append new columns on to the end.
                        //map is going to produce more than 1 output it will be easier to add it to the end
                        ret.removeColumn(ret.columnIndex(retColumn));
                    }
                } else if (column instanceof BooleanColumn) {
                    BooleanColumn booleanColumn = (BooleanColumn) column;
                    BooleanColumn retBooleanColumn = (BooleanColumn) retColumn;
                    if (sameTypesForOutput)
                        for (int i = 0; i < booleanColumn.size(); i++) {
                            retBooleanColumn.set(i, (Boolean) transform.map(booleanColumn.get(i)));
                        }
                    else {
                        //remove the column and append new columns on to the end.
                        //map is going to produce more than 1 output it will be easier to add it to the end
                        ret.removeColumn(ret.columnIndex(retColumn));
                    }
                } else if (column instanceof CategoryColumn) {
                    CategoryColumn categoryColumn = (CategoryColumn) column;
                    CategoryColumn retCategoryColumn = (CategoryColumn) retColumn;
                    if (sameTypesForOutput)
                        for (int i = 0; i < categoryColumn.size(); i++) {
                            retCategoryColumn.set(i, (String) transform.map(categoryColumn.get(i)));
                        }
                    else {
                        //remove the column and append new columns on to the end.
                        //map is going to produce more than 1 output it will be easier to add it to the end
                        ret.removeColumn(ret.columnIndex(retColumn));
                    }
                } else if (column instanceof DateColumn) {
                    DateColumn dateColumn = (DateColumn) column;
                    DateColumn retDateColumn = (DateColumn) retColumn;
                    if (sameTypesForOutput)
                        for (int i = 0; i < dateColumn.size(); i++) {
                            retDateColumn.set(i, (Integer) transform.map(dateColumn.get(i)));
                        }
                    else {
                        //remove the column and append new columns on to the end.
                        //map is going to produce more than 1 output it will be easier to add it to the end
                        ret.removeColumn(ret.columnIndex(retColumn));
                    }
                }

                else if (column instanceof IntColumn) {
                    IntColumn intColumn = (IntColumn) column;
                    IntColumn retIntColumn = (IntColumn) retColumn;
                    if (newColumnNames.length == 1)
                        for (int i = 0; i < intColumn.size(); i++) {
                            retIntColumn.set(i, (Integer) transform.map(intColumn.get(i)));
                        }
                    else {
                        //remove the column and append new columns on to the end.
                        //map is going to produce more than 1 output it will be easier to add it to the end
                        ret.removeColumn(ret.columnIndex(retColumn));
                    }
                } else if (column instanceof ShortColumn) {
                    ShortColumn shortColumn = (ShortColumn) column;
                    ShortColumn retShortColumn = (ShortColumn) retColumn;
                    if (sameTypesForOutput)
                        for (int i = 0; i < shortColumn.size(); i++) {
                            retShortColumn.set(i, (Short) transform.map(shortColumn.get(i)));
                        }
                    else {
                        //remove the column and append new columns on to the end.
                        //map is going to produce more than 1 output it will be easier to add it to the end
                        ret.removeColumn(ret.columnIndex(retColumn));
                    }
                } else {
                    throw new IllegalStateException("Illegal column type " + column.getClass());
                }


            }
        }


        return ret;
    }


    /**
     * Determine the input type based on the column metadata
     * @param row the row of the column to get the input for
     * @param inputColumns the input columns to get input for
     * @return a list of the type for the given metadata
     */
    public static Object determineInput(int row, Column... inputColumns) {
        if (inputColumns.length > 1) {
            switch (inputColumns[0].columnMetadata().getType()) {
                case BOOLEAN:
                    List<Boolean> ret = new ArrayList<>();
                    for (Column c : inputColumns) {
                        BooleanColumn b = (BooleanColumn) c;
                        ret.add(b.get(row));
                    }
                    return ret;
                case FLOAT:
                    List<Float> retFloat = new ArrayList<>();
                    for (Column c : inputColumns) {
                        FloatColumn floats = (FloatColumn) c;
                        retFloat.add(floats.get(row));
                    }
                    return retFloat;
                case INTEGER:
                    List<Integer> integers = new ArrayList<>();
                    for (Column c : inputColumns) {
                        IntColumn intColumn = (IntColumn) c;
                        integers.add(intColumn.get(row));
                    }
                    return integers;
                case LONG_INT:
                    List<Long> longs = new ArrayList<>();
                    for (Column c : inputColumns) {
                        LongColumn longColumn = (LongColumn) c;
                        longs.add(longColumn.get(row));
                    }
                    return longs;
                case CATEGORY:
                    List<String> strings = new ArrayList<>();
                    for (Column c : inputColumns) {
                        CategoryColumn categoryColumn = (CategoryColumn) c;
                        strings.add(categoryColumn.get(row));
                    }
                    return strings;
                default:
                    throw new IllegalStateException(
                                    "Illegal column type " + inputColumns[0].columnMetadata().getType());
            }
        } else {
            return getEntry(inputColumns[0], row);
        }
    }



    /**
     * Set an entry from the given columns
     * @param columns the columns to set the entry for
     * @param row the row to get the entry from
     * @param value an object of type {@link List}
     */
    public static void setEntryList(Column[] columns, int row, Object value) {
        for (Column column : columns) {
            if (column instanceof FloatColumn) {
                List<Float> floatValues = (List<Float>) value;
                for (Float f : floatValues)
                    setEntry(column, row, f);
            } else if (column instanceof LongColumn) {
                List<Long> longValues = (List<Long>) value;
                for (Long l : longValues)
                    setEntry(column, row, l);
            } else if (column instanceof BooleanColumn) {
                List<Boolean> booleanList = (List<Boolean>) value;
                for (Boolean b : booleanList)
                    setEntry(column, row, b);
            } else if (column instanceof CategoryColumn) {
                List<String> stringList = (List<String>) value;
                for (String s : stringList)
                    setEntry(column, row, s);
            } else if (column instanceof DateColumn) {
                List<Integer> integerListDate = (List<Integer>) value;
                for (Integer i : integerListDate)
                    setEntry(column, row, i);
            }

            else if (column instanceof IntColumn) {
                List<Integer> ints = (List<Integer>) value;
                for (Integer i : ints)
                    setEntry(column, row, i);
            } else if (column instanceof ShortColumn) {
                List<Short> shortList = (List<Short>) value;
                for (Short s : shortList)
                    setEntry(column, row, s);
            }


            else {
                throw new IllegalStateException("Illegal column type " + column.getClass());
            }
        }

    }

    /**
     * Get an entry from the given column
     * @param column the column to get the entry from
     * @param row the row to get the entry from
     * @param value the value to set
     * @return the entry from the given column
     * at the given row
     */
    public static void setEntry(Column column, int row, Object value) {
        if (column instanceof FloatColumn) {
            FloatColumn floatColumn = (FloatColumn) column;
            floatColumn.set(row, (float) value);
        } else if (column instanceof LongColumn) {
            LongColumn longColumn = (LongColumn) column;
            longColumn.set(row, (long) value);
        } else if (column instanceof BooleanColumn) {
            BooleanColumn booleanColumn = (BooleanColumn) column;
            booleanColumn.set(row, (boolean) value);
        } else if (column instanceof CategoryColumn) {
            CategoryColumn categoryColumn = (CategoryColumn) column;
            categoryColumn.set(row, value.toString());
        } else if (column instanceof DateColumn) {
            DateColumn dateColumn = (DateColumn) column;
            dateColumn.set(row, (int) value);
        }

        else if (column instanceof IntColumn) {
            IntColumn intColumn = (IntColumn) column;
            intColumn.set(row, (int) value);
        } else if (column instanceof ShortColumn) {
            ShortColumn shortColumn = (ShortColumn) column;
            shortColumn.set(row, (short) value);
        }


        else {
            throw new IllegalStateException("Illegal column type " + column.getClass());
        }
    }

    /**
     * Get an entry from the given column
     * @param column the column to get the entry from
     * @param row the row to get the entry from
     * @return the entry from the given column
     * at the given row
     */
    public static Object getEntry(Column column, int row) {
        if (column instanceof FloatColumn) {
            FloatColumn floatColumn = (FloatColumn) column;
            return floatColumn.get(row);
        } else if (column instanceof LongColumn) {
            LongColumn longColumn = (LongColumn) column;
            return longColumn.get(row);
        } else if (column instanceof BooleanColumn) {
            BooleanColumn booleanColumn = (BooleanColumn) column;
            return booleanColumn.get(row);
        } else if (column instanceof CategoryColumn) {
            CategoryColumn categoryColumn = (CategoryColumn) column;
            return categoryColumn.get(row);
        } else if (column instanceof DateColumn) {
            DateColumn dateColumn = (DateColumn) column;
            return dateColumn.get(row);
        }

        else if (column instanceof IntColumn) {
            IntColumn intColumn = (IntColumn) column;
            return intColumn.get(row);
        } else if (column instanceof ShortColumn) {
            ShortColumn shortColumn = (ShortColumn) column;
            return shortColumn.get(row);
        }


        else {
            throw new IllegalStateException("Illegal column type " + column.getClass());
        }
    }

    /**
     * Create a matrix from a table where
     * the matrix will be of n rows x m columns
     *
     * @param table the table to create
     * @return the matrix created from this table
     */
    public static INDArray arrayFromTable(Table table) {
        INDArray arr = Nd4j.create(table.rowCount(), table.columnCount());
        for (int i = 0; i < table.rowCount(); i++) {
            for (int j = 0; j < table.columnCount(); j++) {
                arr.putScalar(i, j, Double.valueOf(table.get(i, j)));
            }
        }

        return arr;
    }

    /**
     * Convert an all numeric table
     * to a list of records
     * @param table the table to convert
     * @return the list of records from
     * the given table
     */
    public static List<List<Writable>> fromTable(Table table) {
        List<List<Writable>> ret = new ArrayList<>();
        for (int i = 0; i < table.rowCount(); i++) {
            ret.add(new ArrayList<>());
            for (int j = 0; j < table.columnCount(); j++) {
                ret.get(i).add(new DoubleWritable(Double.valueOf(table.get(i, j))));
            }
        }
        return ret;
    }

    /**
     * Create a table from records and
     * a given schema.
     * The records should either be writables such that
     * each list should be interpreted as a row in the table.
     * Optionally, you can also have singleton lists of
     * {@link NDArrayWritable} to represent columns.
     * The given ndarray must be a row of 1 x m where m
     * is schema.numColumns()
     * @param writable the records to create the table from
     * @param schema the schema to use
     * @return the created table
     */
    public static Table fromRecordsAndSchema(List<List<Writable>> writable, Schema schema) {
        Table table = Table.create("table", columnsForSchema(schema));
        for (int i = 0; i < writable.size(); i++) {
            List<Writable> row = writable.get(i);
            if (row.size() == 1 && row.get(0) instanceof NDArrayWritable) {
                NDArrayWritable ndArrayWritable = (NDArrayWritable) row.get(0);
                INDArray arr = ndArrayWritable.get();
                if (arr.columns() != schema.numColumns())
                    throw new IllegalArgumentException("Found ndarray writable of illegal size " + arr.columns());
                for (int j = 0; j < arr.length(); j++) {
                    table.floatColumn(j).append(arr.getDouble(j));
                }
            } else if (row.size() == schema.numColumns()) {
                for (int j = 0; j < row.size(); j++) {
                    table.floatColumn(j).append(row.get(j).toDouble());
                }
            } else
                throw new IllegalArgumentException("Illegal writable list of size " + row.size() + " at index " + i);
        }
        return table;
    }


    /**
     * Create a table from the given schema
     * @param schema the schema to create the table from
     * @return the created table
     */
    public static Table tableFromSchema(Schema schema) {
        return Table.create("newTable", columnsForSchema(schema));
    }

    /**
     * Extract a column array from the given schema
     * @param schema the schema to get columns from
     * @return a column array based on the given schema
     */
    public static Column[] columnsForSchema(Schema schema) {
        Column[] ret = new Column[schema.numColumns()];
        for (int i = 0; i < schema.numColumns(); i++) {
            switch (schema.getType(i)) {
                case Double:
                    ret[i] = new FloatColumn(schema.getName(i));
                    break;
                case Float:
                    ret[i] = new FloatColumn(schema.getName(i));
                    break;
                case Long:
                    ret[i] = new LongColumn(schema.getName(i));
                    break;
                case Integer:
                    ret[i] = new IntColumn(schema.getName(i));
                    break;
                case Categorical:
                    ret[i] = new CategoryColumn(schema.getName(i), 4);
                    break;
                case Time:
                    ret[i] = new DateColumn(new ColumnMetadata(new LongColumn(schema.getName(i))));
                    break;
                case Boolean:
                    ret[i] = new BooleanColumn(new ColumnMetadata(new IntColumn(schema.getName(i))));
                    break;
            }
        }
        return ret;
    }



}
