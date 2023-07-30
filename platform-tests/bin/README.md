# Java configuration for surefire

The "java" file here is actually a shell script we use
to allow us to customize surefire test execution
via the <jvm> parameter in surefire.

Surefire "detects" java by checking for a parent bin directory
and a java executable. There is no configurable way
to pass a wrapper script. Thus we do this.