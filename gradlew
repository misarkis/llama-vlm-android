#!/bin/sh
# Gradle wrapper script for Unix systems

# Determine the project root directory
APP_BASE_NAME=`basename "$0"`
APP_HOME="`pwd -P`"

# Find Java (use Android Studio bundled JBR)
if [ -f "/c/Program Files/Android/Android Studio/jbr/bin/java.exe" ]; then
    JAVA="/c/Program Files/Android/Android Studio/jbr/bin/java.exe"
else
    JAVA="java"
fi

# Find the gradle wrapper jar
WRAPPER_JAR="$APP_HOME/gradle/wrapper/gradle-wrapper.jar"

# Execute Gradle
exec "$JAVA" -cp "$WRAPPER_JAR" org.gradle.wrapper.GradleWrapperMain "$@"
