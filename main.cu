#include <CppUTest/CommandLineTestRunner.h>
/*
#include <shapetests.h>
#include <teststring.h>
#include <transformtests.h>
#include <broadcasttests.h>
#include<indexreducetests.h>
#include<pairwise_transform_tests.h>
#include<reduce3tests.h>
#include<reducetests.h>
#include <scalartests.h>
*/

/*
#include <shapetests.h>
#include <scalartests.h>
#include <transformtests.h>
#include <pairwise_transform_tests.h>
#include <broadcaststests.h>
#include <reducetests.h>
#include <reduce3tests.h>
#include <indexreducetests.h>
*/
#include <transformtests.h>

int main(int ac, char** av) {
	return CommandLineTestRunner::RunAllTests(ac, av);
}
//IMPORT_TEST_GROUP(ScalarTransform);
//IMPORT_TEST_GROUP(BroadCasting);
//IMPORT_TEST_GROUP(PairWiseTransform);
//IMPORT_TEST_GROUP(Shape);
IMPORT_TEST_GROUP(Transform);
//IMPORT_TEST_GROUP(Reduce);
//IMPORT_TEST_GROUP(Reduce3);
//IMPORT_TEST_GROUP(IndexReduce);


