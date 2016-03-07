/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// These are helper functions for the SDK samples (string parsing, timers, etc)
#ifndef STRING_HELPER_H
#define STRING_HELPER_H
#include <dll.h>

//#include <stdio.h>
//#include <stdlib.h>
//#include <fstream>
//#include <string>

#ifdef _WIN32
#ifndef STRCASECMP
#define STRCASECMP  _stricmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif
#ifndef STRCPY
#define STRCPY(sFilePath, nLength, sPath) strcpy_s(sFilePath, nLength, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result != 0)
#endif
#ifndef SSCANF
#define SSCANF sscanf_s
#endif

#else

//#include <string.h>
//#include <strings.h>
#include <dll.h>

#ifndef STRCASECMP
#define STRCASECMP  strcasecmp
#endif
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif
#ifndef STRCPY
#define STRCPY(sFilePath, nLength, sPath) strcpy(sFilePath, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) (fHandle = fopen(filename, mode))
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result == NULL)
#endif
#ifndef SSCANF
#define SSCANF sscanf
#endif
#endif

// CUDA Utility Helper Functions
inline int stringRemoveDelimiter(char delimiter, const char *string) {
	int string_start = 0;

	while (string[string_start] == delimiter) {
		string_start++;
	}

	if (string_start >= (int) strlen(string) - 1) {
		return 0;
	}

	return string_start;
}

inline int getFileExtension(char *filename, char **extension) {
	int string_length = (int) strlen(filename);

	while (filename[string_length--] != '.') {
		if (string_length == 0)
			break;
	}
	if (string_length > 0)
		string_length += 2;

	if (string_length == 0)
		*extension = NULL;
	else
		*extension = &filename[string_length];

	return string_length;
}

inline int checkCmdLineFlag(const int argc, const char **argv,
		const char *string_ref) {
	bool bFound = false;

	if (argc >= 1) {
		for (int i = 1; i < argc; i++) {
			int string_start = stringRemoveDelimiter('-', argv[i]);
			const char *string_argv = &argv[i][string_start];

			const char *equal_pos = strchr(string_argv, '=');
			int argv_length = (int) (
					equal_pos == 0 ?
							strlen(string_argv) : equal_pos - string_argv);

			int length = (int) strlen(string_ref);

			if (length == argv_length
					&& !STRNCASECMP(string_argv, string_ref, length)) {

				bFound = true;
				continue;
			}
		}
	}

	return (int) bFound;
}

inline int getCmdLineArgumentInt(const int argc, const char **argv,
		const char *string_ref) {
	bool bFound = false;
	int value = -1;

	if (argc >= 1) {
		for (int i = 1; i < argc; i++) {
			int string_start = stringRemoveDelimiter('-', argv[i]);
			const char *string_argv = &argv[i][string_start];
			int length = (int) strlen(string_ref);

			if (!STRNCASECMP(string_argv, string_ref, length)) {
				if (length + 1 <= (int) strlen(string_argv)) {
					int auto_inc = (string_argv[length] == '=') ? 1 : 0;
					value = atoi(&string_argv[length + auto_inc]);
				} else {
					value = 0;
				}

				bFound = true;
				continue;
			}
		}
	}

	if (bFound) {
		return value;
	} else {
		return 0;
	}
}

inline float getCmdLineArgumentFloat(const int argc, const char **argv,
		const char *string_ref) {
	bool bFound = false;
	float value = -1;

	if (argc >= 1) {
		for (int i = 1; i < argc; i++) {
			int string_start = stringRemoveDelimiter('-', argv[i]);
			const char *string_argv = &argv[i][string_start];
			int length = (int) strlen(string_ref);

			if (!STRNCASECMP(string_argv, string_ref, length)) {
				if (length + 1 <= (int) strlen(string_argv)) {
					int auto_inc = (string_argv[length] == '=') ? 1 : 0;
					value = (float) atof(&string_argv[length + auto_inc]);
				} else {
					value = 0.f;
				}

				bFound = true;
				continue;
			}
		}
	}

	if (bFound) {
		return value;
	} else {
		return 0;
	}
}

inline bool getCmdLineArgumentString(const int argc, const char **argv,
		const char *string_ref, char **string_retval) {
	bool bFound = false;

	if (argc >= 1) {
		for (int i = 1; i < argc; i++) {
			int string_start = stringRemoveDelimiter('-', argv[i]);
			char *string_argv = (char *) &argv[i][string_start];
			int length = (int) strlen(string_ref);

			if (!STRNCASECMP(string_argv, string_ref, length)) {
				*string_retval = &string_argv[length + 1];
				bFound = true;
				continue;
			}
		}
	}

	if (!bFound) {
		*string_retval = NULL;
	}

	return bFound;
}

#endif
