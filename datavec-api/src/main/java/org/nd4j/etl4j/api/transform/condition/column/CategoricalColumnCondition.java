package org.nd4j.etl4j.api.transform.condition.column;

import org.nd4j.etl4j.api.transform.condition.SequenceConditionMode;
import org.nd4j.etl4j.api.writable.Writable;
import org.nd4j.etl4j.api.transform.condition.ConditionOp;

import java.util.Set;

/**
 * Condition that applies to the values in a Categorical column, using a {@link ConditionOp}
 *
 * @author Alex Black
 */
public class CategoricalColumnCondition extends BaseColumnCondition {

    private final ConditionOp op;
    private final String value;
    private final Set<String> set;

    /**
     * Constructor for conditions equal or not equal.
     * Uses default sequence condition mode, {@link BaseColumnCondition#DEFAULT_SEQUENCE_CONDITION_MODE}
     *
     * @param column Column to check for the condition
     * @param op     Operation (== or != only)
     * @param value  Value to use in the condition
     */
    public CategoricalColumnCondition(String column, ConditionOp op, String value) {
        this(column, DEFAULT_SEQUENCE_CONDITION_MODE, op, value);
    }

    /**
     * Constructor for conditions equal or not equal
     *
     * @param column                Column to check for the condition
     * @param sequenceConditionMode Mode for handling sequence data
     * @param op                    Operation (== or != only)
     * @param value                 Value to use in the condition
     */
    public CategoricalColumnCondition(String column, SequenceConditionMode sequenceConditionMode,
                                      ConditionOp op, String value) {
        super(column, sequenceConditionMode);
        if (op != ConditionOp.Equal && op != ConditionOp.NotEqual) {
            throw new IllegalArgumentException("Invalid condition op: can only use this constructor with Equal or NotEqual conditions");
        }
        this.op = op;
        this.value = value;
        this.set = null;
    }


    /**
     * Constructor for operations: ConditionOp.InSet, ConditionOp.NotInSet
     * Uses default sequence condition mode, {@link BaseColumnCondition#DEFAULT_SEQUENCE_CONDITION_MODE}
     *
     * @param column Column to check for the condition
     * @param op     Operation. Must be either ConditionOp.InSet, ConditionOp.NotInSet
     * @param set    Set to use in the condition
     */
    public CategoricalColumnCondition(String column, ConditionOp op, Set<String> set) {
        this(column, DEFAULT_SEQUENCE_CONDITION_MODE, op, set);
    }

    /**
     * Constructor for operations: ConditionOp.InSet, ConditionOp.NotInSet
     *
     * @param column                Column to check for the condition
     * @param sequenceConditionMode Mode for handling sequence data
     * @param op                    Operation. Must be either ConditionOp.InSet, ConditionOp.NotInSet
     * @param set                   Set to use in the condition
     */
    public CategoricalColumnCondition(String column, SequenceConditionMode sequenceConditionMode,
                                      ConditionOp op, Set<String> set) {
        super(column, sequenceConditionMode);
        if (op != ConditionOp.InSet && op != ConditionOp.NotInSet) {
            throw new IllegalArgumentException("Invalid condition op: can ONLY use this constructor with InSet or NotInSet ops");
        }
        this.op = op;
        this.value = null;
        this.set = set;
    }


    @Override
    public boolean columnCondition(Writable writable) {
        switch (op) {
            case Equal:
                return value.equals(writable.toString());
            case NotEqual:
                return !value.equals(writable.toString());
            case InSet:
                return set.contains(writable.toString());
            case NotInSet:
                return !set.contains(writable.toString());
            case LessThan:
            case LessOrEqual:
            case GreaterThan:
            case GreaterOrEqual:
                throw new UnsupportedOperationException("Cannot use ConditionOp \"" + op + "\" on Categorical column");
            default:
                throw new RuntimeException("Unknown or not implemented op: " + op);
        }
    }
}
