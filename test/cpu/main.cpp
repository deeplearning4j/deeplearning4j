#include <CppUTest/CommandLineTestRunner.h>
//#include <pairwise_transform_tests.h>
//#include <scalartests.h>
//#include <transformtests.h>
//#include <broadcasttests.h>
//#include <shapetests.h>
//#include <reducetests.h>
//#include <reduce3tests.h>
#include <indexreducetests.h>
#include <summarystatsreducetest.h>

int main(int ac, char** av) {
#ifdef __CUDACC__
	cudaDeviceSetLimit(cudaLimitStackSize,20000);
#endif
	return CommandLineTestRunner::RunAllTests(ac, av);
}

//IMPORT_TEST_GROUP(PairWiseTransform);
//IMPORT_TEST_GROUP(ScalarTransform);
//IMPORT_TEST_GROUP(Transform);
//IMPORT_TEST_GROUP(BroadCasting);
//IMPORT_TEST_GROUP(Shape);
//IMPORT_TEST_GROUP(Reduce);
//IMPORT_TEST_GROUP(Reduce3);
IMPORT_TEST_GROUP(IndexReduce);
//IMPORT_TEST_GROUP(SummaryStatsReduce);

