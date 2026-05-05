Generate or run tests for recent changes. $ARGUMENTS

Follow these steps:
1. Run `git diff` to identify changed files (or `git diff HEAD~1` if no uncommitted changes)
2. Identify the testing framework used in this project:
   - Look for existing test files near the changed code
   - Check build config (pom.xml, package.json, etc.) for test dependencies
3. Read the changed files and existing tests
4. If args say "run": execute the existing tests for the changed modules
5. If args say "generate" or no specific instruction:
   - Generate test cases covering the changed code paths
   - Follow the project's existing test conventions
   - Include both happy path and edge case tests
   - Write tests to the appropriate test directory
6. Run the tests to verify they pass

Match existing test style and conventions. Don't over-test simple getters/setters.
