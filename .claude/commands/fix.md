Fix build or test failures. $ARGUMENTS

Follow these steps:
1. If args describe the error, start there
2. Otherwise, try to reproduce the failure:
   - Run the build command (check pom.xml, package.json, Makefile, etc.)
   - Run the test suite
3. Read the error output carefully:
   - Identify the failing file and line number
   - Understand the error message
4. Read the relevant source files
5. Diagnose the root cause (don't just suppress the error)
6. Apply the fix using edit tools
7. Re-run the build/tests to verify the fix works
8. If the fix introduced new warnings, address them too

Focus on fixing the root cause, not symptoms. If multiple failures exist,
fix them one at a time, re-running tests after each fix.
