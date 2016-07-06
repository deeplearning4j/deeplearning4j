package org.nd4j.etl4j.api.transform.condition.column;

import org.nd4j.etl4j.api.transform.condition.SequenceConditionMode;
import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.condition.ConditionOp;

import java.util.Set;

/**
 * Condition that applies to the values in a Long column, using a {@link ConditionOp}
 *
 * @author Alex Black
 */
public class LongColumnCondition extends BaseColumnCondition {

    private final ConditionOp op;
    private final long value;
    private final Set<Long> set;

    /**
     * Constructor for operations such as less than, equal to, greater than, etc.
     * Uses default sequence condition mode, {@link BaseColumnCondition#DEFAULT_SEQUENCE_CONDITION_MODE}
     *
     * @param column Column to check for the condition
     * @param op     Operation (<, >=, !=, etc)
     * @param value  Value to use in the condition
     */
    public LongColumnCondition(String column, ConditionOp op, long value) {
        this(column, DEFAULT_SEQUENCE_CONDITION_MODE, op, value);
    }

    /**
     * Constructor for operations such as less than, equal to, greater than, etc.
     *
     * @param column                Column to check for the condition
     * @param sequenceConditionMode Mode for handling sequence data
     * @param op                    Operation (<, >=, !=, etc)
     * @param value                 Value to use in the condition
     */
    public LongColumnCondition(String column, SequenceConditionMode sequenceConditionMode,
                               ConditionOp op, long value) {
        super(column, sequenceConditionMode);
        if (op == ConditionOp.InSet || op == ConditionOp.NotInSet) {
            throw new IllegalArgumentException("Invalid condition op: cannot use this constructor with InSet or NotInSet ops");
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
    public LongColumnCondition(String column, ConditionOp op, Set<Long> set) {
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
    public LongColumnCondition(String column, SequenceConditionMode sequenceConditionMode,
                               ConditionOp op, Set<Long> set) {
        super(column, sequenceConditionMode);
        if (op != ConditionOp.InSet && op != ConditionOp.NotInSet) {
            throw new IllegalArgumentException("Invalid condition op: can ONLY use this constructor with InSet or NotInSet ops");
        }
        this.op = op;
        this.value = 0;
        this.set = set;
    }


    @Override
    public boolean columnCondition(Writable writable) {
        switch (op) {
            case LessThan:
                return writable.toLong() < value;
            case LessOrEqual:
                return writable.toLong() <= value;
            case GreaterThan:
                return writable.toLong() > value;
            case GreaterOrEqual:
                return writable.toLong() >= value;
            case Equal:
                return writable.toLong() == value;
            case NotEqual:
                return writable.toLong() != value;
            case InSet:
                return set.contains(writable.toLong());
            case NotInSet:
                return !set.contains(writable.toLong());
            default:
                throw new RuntimeException("Unknown or not implemented op: " + op);
        }
    }
}
