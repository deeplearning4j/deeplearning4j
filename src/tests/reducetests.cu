#include <CppUTest/TestHarness.h>
#include "reducetests.h"

TEST_GROUP(Mean)
{

	static int output_method(const char* output, ...)
	{
		va_list arguments;
		va_start(arguments, output);
		va_end(arguments);
		return 1;
	}
	void setup()
	{

		double *data = (double *) malloc(sizeof(double) * 4);
		for(int i = 0; i < 4; i++) {
			data[i] = i + 1;
		}
		int rank = 2;
		int *shape = (int *) malloc(sizeof(int) * rank);
		shape[0] = 2;
		shape[1] = 2;

		int *stride = (int *)malloc(sizeof(int) * rank);
		stride[0] = 1;
		stride[1] = 2;
	}
	void teardown()
	{
	}
};

TEST(Mean, PrintOk)
{
	testMean();

}
