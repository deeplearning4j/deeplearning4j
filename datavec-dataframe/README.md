DataVec-Tablesaw
================
   
Note: This is a fork of [tablesaw](https://github.com/lwhite1/tablesaw)

focused on integration with nd4j and deeplearning4j. The main focus of this module is going to be
ONLY having the dataframe component. We will be using this as a backend for datavec itself
for local transformations. This also gives developers an alternative api for fairly simple problems.
   
### Tablesaw features: 

#### Data processing & transformation
* Import data from RDBMS and CSV files, local or remote (http, S3, etc.)
* Combine files
* Add and remove columns
* Sort, Group, Filter 
* Map/Reduce operations


#From the original readme:


### Current performance:
You can load a 500,000,000 row, 4 column csv file (35GB on disk) entirely into about 10 GB of memory. If it's in Tablesaw's .saw format, you can load it in 22 seconds. You can query that table in 1-2 ms: fast enough to use as a cache for a Web app.


### Easy to Use is Easy to Say
The goal in this example is to identify the production shifts with the worst performance. These few lines demonstrate __data import__, column-wise operations (__differenceInSeconds()__), filters (__isInQ2()__) grouping and aggegating (__median()__ and __.by()__), and (__top(n)__) calculations. 

```java
    Table ops = Table.createFromCsv("data/operations.csv");                             // load data
    DateTimeColumn start = ops.dateColumn("Date").atTime(ops.timeColumn("Start"));
    DateTimeColumn end = ops.dateColumn("Date").atTime(ops.timeColumn("End");
    LongColumn duration = start.differenceInSeconds(end);                        // calc duration
    duration.setName("Duration");
    ops.addColumn(duration);
    
    Table filtered = ops.selectWhere(                                            // filter
          allOf
              (column("date").isInQ2(),
              (column("SKU").startsWith("429")),
              (column("Operation").isEqualTo("Assembly"))));
   
    Table summary = filtered.median("Duration").by("Facility", "Shift");         // group medians
    FloatArrayList tops = summary.floatColumn("Median").top(5);                  // get "slowest"

```
