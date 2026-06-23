// Top-level build file
buildscript {
    repositories {
        google()
        mavenCentral()
        flatDir {
            dirs("$rootDir/libs")
        }
    }
}

allprojects {
    repositories {
        google()
        mavenCentral()
        flatDir {
            dirs("$rootDir/libs")
        }
    }
}

plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
}
