package org.datavec.api.util.jdbc;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Iterator;
import org.apache.commons.dbutils.ResultSetIterator;

/**
 * Encapsulation of ResultSetIterator to allow resetting
 *
 * @author Adrien Plagnol
 */
public class ResettableResultSetIterator implements Iterator<Object[]> {

    private ResultSet rs;
    private ResultSetIterator base;

    public ResettableResultSetIterator(ResultSet rs) {
        this.rs = rs;
        this.base = new ResultSetIterator(rs);
    }

    public void reset() {
        try {
            this.rs.beforeFirst();
        } catch (SQLException e) {
            throw new RuntimeException("Could not reset ResultSetIterator", e);
        }
    }

    @Override
    public boolean hasNext() {
        return this.base.hasNext();
    }

    @Override
    public Object[] next() {
        return base.next();
    }

    @Override
    public void remove() {
        base.remove();
    }
}
