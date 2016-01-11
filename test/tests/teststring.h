/*
 * teststring.h
 *
 *  Created on: Jan 9, 2016
 *      Author: agibsonccc
 */

#ifndef TESTSTRING_H_
#define TESTSTRING_H_
#include <op.h>
#include "testhelpers.h"



TEST_GROUP(String) {
	static int output_method(const char* output, ...) {
		va_list arguments;
		va_start(arguments, output);
		va_end(arguments);
		return 1;
	}
	void setup() {

	}
	void teardown() {
	}
};

TEST(String,StrCmp) {
	int output = functions::ops::strcmp("sigmoid","sigmoid");
	CHECK(output == 0);
}

#endif /* TESTSTRING_H_ */
