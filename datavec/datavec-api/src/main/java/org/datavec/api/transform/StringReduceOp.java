package org.datavec.api.transform;

/**
 * A string reduce op is used for combining strings such as
 * merging, appending, or prepending.
 *
 * The following ops are supported:
 * PREPEND: prepend the first string to the second
 * APPEND: append the first string to the second
 * MERGE: Merge the 2 strings
 * FORMAT: Apply the format (the first column, to the string separated list via csv)
 * @author Adam Gibson
 */
public enum StringReduceOp {
    PREPEND, APPEND, MERGE, REPLACE
}
