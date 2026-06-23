@rem Gradle wrapper script for Windows
@setlocal
@set GRADLE_HOME=%~dp0
@set WRAPPER_JAR=%GRADLE_HOME%gradle\wrapper\gradle-wrapper.jar
@set JAVA_FOUND=0

@rem 1. Check JAVA_HOME environment variable
@if defined JAVA_HOME (
    @set "JAVA_EXE=%JAVA_HOME%\bin\java.exe"
    @if exist "%JAVA_EXE%" (
        "%JAVA_EXE%" -version >nul 2>&1
        @if %ERRORLEVEL%==0 (
            @set "JAVA=%JAVA_EXE%"
            @set JAVA_FOUND=1
        )
    )
)

@rem 2. Check user-local Android Studio
@if %JAVA_FOUND%==0 (
    @set "JAVA_EXE=%LOCALAPPDATA%\Programs\Android\Android Studio\jbr\bin\java.exe"
    @if exist "%JAVA_EXE%" (
        "%JAVA_EXE%" -version >nul 2>&1
        @if %ERRORLEVEL%==0 (
            @set "JAVA=%JAVA_EXE%"
            @set JAVA_FOUND=1
        )
    )
)

@rem 3. Check Program Files Android Studio
@if %JAVA_FOUND%==0 (
    @set "JAVA_EXE=C:\Program Files\Android\Android Studio\jbr\bin\java.exe"
    @if exist "%JAVA_EXE%" (
        "%JAVA_EXE%" -version >nul 2>&1
        @if %ERRORLEVEL%==0 (
            @set "JAVA=%JAVA_EXE%"
            @set JAVA_FOUND=1
        )
    )
)

@rem 4. Check older jre location
@if %JAVA_FOUND%==0 (
    @set "JAVA_EXE=C:\Program Files\Android\Android Studio\jre\bin\java.exe"
    @if exist "%JAVA_EXE%" (
        "%JAVA_EXE%" -version >nul 2>&1
        @if %ERRORLEVEL%==0 (
            @set "JAVA=%JAVA_EXE%"
            @set JAVA_FOUND=1
        )
    )
)

@rem 5. Fall back to system java
@if %JAVA_FOUND%==0 @set JAVA=java

"%JAVA%" -cp "%WRAPPER_JAR%" org.gradle.wrapper.GradleWrapperMain %*
