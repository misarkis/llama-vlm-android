# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in C:/Users/misar/AppData/Local/Android/Sdk/tools/proguard/proguard-android.txt

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep JNI methods
-keep class com.misar.vlmanalyze.** { *; }
