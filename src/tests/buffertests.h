__locale_struct/*
 * buffertests.h
 *
 *  Created on: Jan 1, 2016
 *      Author: agibsonccc
 */

#ifndef BUFFERTESTS_H_
#define BUFFERTESTS_H_
#include <CppUTest/TestHarness.h>
#include <buffer.h>



TEST_GROUP(Buffer)
{

	static int output_method(const char* output, ...) {
		va_list arguments;
		va_start(arguments, output);
		va_end(arguments);
		return 1;
	}
	void setup() {


	}
	void teardown() 	{
	}
};

TEST(Buffer, SimpleTwoByTwo)
{
/*
	nd4j::buffer::Buffer<double> *buff;
	int length = 4;
	double *data = new double[length];
	buff = nd4j::buffer::createBuffer<double>(data,length);
	for(int i = 0; i < length; i++) {
		buff[i] = (double) i;
	}
	nd4j::buffer::freeBuffer<double>(&buff);

*/

	CHECK(true);
}






#endif /* BUFFERTESTS_H_ */
