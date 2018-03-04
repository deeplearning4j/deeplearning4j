package org.deeplearning4j.spark.impl.common.repartition;

import com.google.common.base.Predicate;
import com.google.common.collect.Collections2;
import org.apache.spark.Partitioner;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

/**
 * This is a custom partitioner that rebalances a minimum of elements
 * it expects a key in the form (SparkUID, class)
 *
 * @author huitseeker
 */
public class HashingBalancedPartitioner extends Partitioner {
    private final int numClasses; // Total number of element classes
    private final int numPartitions; // Total number of partitions
    // partitionWeightsByClass : numClasses lists of numPartitions elements
    // where each element is the partition's relative share of its # of elements w.r.t the per-partition mean
    // e.g. we have 3 partitions, with red and blue elements, red is indexed by 0, blue by 1:
    //  [ r, r, r, r, b, b, b ], [r, b, b], [b, b, b, b, b, r, r]
    // avg # red elems per partition : 2.33
    // avg # blue elems per partition : 3.33
    // partitionWeightsByClass = [[1.714, .429, .857], [0.9, 0.6, 1.5]]
    private List<List<Double>> partitionWeightsByClass;

    // The cumulative distribution of jump probabilities of extra elements by partition, by class
    // 0 for partitions that already have enough elements
    private List<List<Double>> jumpTable;
    private Random r;

    public HashingBalancedPartitioner(List<List<Double>> partitionWeightsByClass) {
        List<List<Double>> pw = checkNotNull(partitionWeightsByClass);
        checkArgument(!pw.isEmpty(), "Partition weights are required");
        checkArgument(pw.size() >= 1, "There should be at least one element class");
        checkArgument(!checkNotNull(pw.get(0)).isEmpty(), "At least one partition is required");
        this.numClasses = pw.size();
        this.numPartitions = pw.get(0).size();
        for (int i = 1; i < pw.size(); i++) {
            checkArgument(checkNotNull(pw.get(i)).size() == this.numPartitions,
                            "Non-consistent partition weight specification");
            // you also should have sum(pw.get(i)) = this.numPartitions
        }
        this.partitionWeightsByClass = partitionWeightsByClass; // p_(j, i)

        List<List<Double>> jumpsByClass = new ArrayList<>();;
        for (int j = 0; j < numClasses; j++) {
            Double totalImbalance = 0D; // i_j = sum(max(1 - p_(j, i), 0) , i = 1..numPartitions)
            for (int i = 0; i < numPartitions; i++) {
                totalImbalance += partitionWeightsByClass.get(j).get(i) >= 0
                                ? Math.max(1 - partitionWeightsByClass.get(j).get(i), 0) : 0;
            }
            Double sumProb = 0D;
            List<Double> cumulProbsThisClass = new ArrayList<>();
            for (int i = 0; i < numPartitions; i++) {
                if (partitionWeightsByClass.get(j).get(i) >= 0 && (totalImbalance > 0 || sumProb >= 1)) {
                    Double thisPartitionRelProb =
                                    Math.max(1 - partitionWeightsByClass.get(j).get(i), 0) / totalImbalance;
                    if (thisPartitionRelProb > 0) {
                        sumProb += thisPartitionRelProb;
                        cumulProbsThisClass.add(sumProb);
                    } else {
                        cumulProbsThisClass.add(0D);
                    }
                } else {
                    // There's no more imbalance, every jumpProb is > 1
                    cumulProbsThisClass.add(0D);
                }
            }
            jumpsByClass.add(cumulProbsThisClass);
        }

        this.jumpTable = jumpsByClass;
    }

    @Override
    public int numPartitions() {
        return Collections2.filter(partitionWeightsByClass.get(0), new Predicate<Double>() {
            @Override
            public boolean apply(Double aDouble) {
                return aDouble >= 0;
            }
        }).size();
    }

    @Override
    public int getPartition(Object key) {
        checkArgument(key instanceof Tuple2, "The key should be in the form: Tuple2(SparkUID, class) ...");
        Tuple2<Long, Integer> uidNclass = (Tuple2<Long, Integer>) key;
        Long uid = uidNclass._1();
        Integer partitionId = (int) (uid % numPartitions);
        Integer elementClass = uidNclass._2();

        Double jumpProbability = Math.max(1D - 1D / partitionWeightsByClass.get(elementClass).get(partitionId), 0D);
        LinearCongruentialGenerator rand = new LinearCongruentialGenerator(uid);

        Double thisJumps = rand.nextDouble();
        Integer thisPartition = partitionId;
        if (thisJumps < jumpProbability) {
            // Where do we jump ?
            List<Double> jumpsTo = jumpTable.get(elementClass);
            Double destination = rand.nextDouble();
            Integer probe = 0;

            while (jumpsTo.get(probe) < destination) {
                probe++;
            }
            thisPartition = probe;
        }

        return thisPartition;
    }

    // Multiplier chosen for nice distribution properties when successive random values are used to form tuples,
    // which is the case with Spark's uid.
    // See P. L'Ecuyer. Tables of Linear Congruential Generators of Different Sizes and Good Lattice
    // Structure. In Mathematics of Computation 68 (225): pages 249–260
    //
    static final class LinearCongruentialGenerator {
        private long state;

        public LinearCongruentialGenerator(long seed) {
            this.state = seed;
        }

        public double nextDouble() {
            state = 2862933555777941757L * state + 1;
            return ((double) ((int) (state >>> 33) + 1)) / (0x1.0p31);
        }
    }
}
