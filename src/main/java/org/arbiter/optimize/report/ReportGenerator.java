package org.arbiter.optimize.report;

import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.api.saving.ResultReference;
import org.arbiter.optimize.exceptions.ReportException;

import java.util.Collection;

public interface ReportGenerator<T,M,A> {

    /** Generate a report, given all result references */
    void generateReport(Collection<ResultReference<T,M,A>> resultReferences) throws ReportException;

    /** Generate a report, given the OptimizationResult objects */
    void generateReportFromResults(Collection<OptimizationResult<T,M,A>> results) throws ReportException;

}
