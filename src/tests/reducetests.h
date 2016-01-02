/*
 * reducetests.h
 *
 *  Created on: Dec 31, 2015
 *      Author: agibsonccc
 */

#ifndef REDUCETESTS_H_
#define REDUCETESTS_H_
#include <array.h>
#include <CppUTest/TestHarness.h>


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

	}
	void teardown()
	{
	}
};

TEST(Mean, SimpleTwoByTwo)
{


}




#endif /* REDUCETESTS_H_ */
