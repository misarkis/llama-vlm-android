# =============================================================================
# VLM-Analyze Android - Setup Configuration (MSYS2 UCRT64)
# =============================================================================
# REQUIRED: All dependencies must be installed for the app to function.
# Run this script ONCE after cloning the repository to configure your environment.
# This creates a .env file that all build scripts will automatically load.
#
# Usage: source ./setup-config.sh
# =============================================================================

echo "=== VLM-Analyze Android Setup ==="
echo "All dependencies are REQUIRED for on-device inference."
echo ""

missing=()

# -----------------------------------------------------------------------------
# MSYS2 UCRT64 - REQUIRED
# -----------------------------------------------------------------------------
echo "1. MSYS2 UCRT64 (REQUIRED - for building llama.cpp):"
if [ -n "$MSYS2_PATH" ]; then
    echo "   Using existing: $MSYS2_PATH"
else
    DEFAULT_MSYS2="/c/msys64"
    echo "   Default: $DEFAULT_MSYS2"
    read -p "   Enter path (press Enter for default): " MSYS2_PATH
    MSYS2_PATH="${MSYS2_PATH:-$DEFAULT_MSYS2}"
fi

if [ ! -d "$MSYS2_PATH" ]; then
    missing+=("MSYS2 not found: $MSYS2_PATH")
elif [ ! -f "$MSYS2_PATH/ucrt64/bin/gcc.exe" ]; then
    missing+=("MSYS2 UCRT64 toolchain not found. Run: pacman -S mingw-w64-ucrt-x86_64-toolchain")
else
    echo "   [OK] Set: $MSYS2_PATH"
fi
echo ""

# -----------------------------------------------------------------------------
# ANDROID SDK - REQUIRED
# -----------------------------------------------------------------------------
echo "2. Android SDK (REQUIRED):"
if [ -n "$ANDROID_SDK_ROOT" ]; then
    echo "   Using existing: $ANDROID_SDK_ROOT"
else
    DEFAULT_SDK="/c/Users/$USER/AppData/Local/Android/Sdk"
    echo "   Default: $DEFAULT_SDK"
    read -p "   Enter path (press Enter for default): " ANDROID_SDK_ROOT
    ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$DEFAULT_SDK}"
fi
export ANDROID_HOME="$ANDROID_SDK_ROOT"
export PATH="$ANDROID_SDK_ROOT/platform-tools:$PATH"

if [ ! -d "$ANDROID_SDK_ROOT" ]; then
    missing+=("Android SDK not found: $ANDROID_SDK_ROOT")
else
    echo "   [OK] Set: $ANDROID_SDK_ROOT"
fi
echo ""

# -----------------------------------------------------------------------------
# ANDROID NDK - REQUIRED
# -----------------------------------------------------------------------------
echo "3. Android NDK (REQUIRED):"
if [ -n "$ANDROID_NDK_HOME" ]; then
    echo "   Using existing: $ANDROID_NDK_HOME"
else
    DEFAULT_NDK="$ANDROID_SDK_ROOT/ndk/25.1.8937393"
    echo "   Default: $DEFAULT_NDK"
    read -p "   Enter path (press Enter for default): " ANDROID_NDK_HOME
    ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$DEFAULT_NDK}"
fi

if [ ! -d "$ANDROID_NDK_HOME" ]; then
    missing+=("Android NDK not found: $ANDROID_NDK_HOME")
else
    echo "   [OK] Set: $ANDROID_NDK_HOME"
fi
echo ""

# -----------------------------------------------------------------------------
# JAVA JDK - REQUIRED
# -----------------------------------------------------------------------------
echo "4. Java JDK 17+ (REQUIRED):"
if [ -n "$JAVA_HOME" ]; then
    echo "   Using existing: $JAVA_HOME"
else
    DEFAULT_JDK="/c/Program\ Files/Android/Android\ Studio/jbr"
    echo "   Default (Android Studio JDK): $DEFAULT_JDK"
    read -p "   Enter path (press Enter for default): " JAVA_HOME
    JAVA_HOME="${JAVA_HOME:-$DEFAULT_JDK}"
fi
export PATH="$JAVA_HOME/bin:$PATH"

if [ ! -d "$JAVA_HOME" ]; then
    missing+=("Java JDK not found: $JAVA_HOME")
else
    echo "   [OK] Set: $JAVA_HOME"
fi
echo ""

# -----------------------------------------------------------------------------
# HEXAGON SDK - REQUIRED
# -----------------------------------------------------------------------------
echo "5. Hexagon SDK (REQUIRED - for NPU/HTP acceleration):"
if [ -n "$HEXAGON_SDK_ROOT" ]; then
    echo "   Using existing: $HEXAGON_SDK_ROOT"
else
    DEFAULT_HEX="/c/Hexagon_SDK/6.6.0.0"
    echo "   Default: $DEFAULT_HEX"
    read -p "   Enter path (press Enter for default): " HEXAGON_SDK_ROOT
    HEXAGON_SDK_ROOT="${HEXAGON_SDK_ROOT:-$DEFAULT_HEX}"
fi

if [ -z "$HEXAGON_SDK_ROOT" ]; then
    missing+=("Hexagon SDK path is REQUIRED for NPU support")
elif [ ! -d "$HEXAGON_SDK_ROOT" ]; then
    missing+=("Hexagon SDK not found: $HEXAGON_SDK_ROOT")
else
    echo "   [OK] Set: $HEXAGON_SDK_ROOT"
fi
echo ""

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
echo "=== Validation ==="
if [ ${#missing[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: Missing required dependencies:"
    for err in "${missing[@]}"; do
        echo "  [X] $err"
    done
    echo ""
    echo "Please fix the paths above before continuing."
    echo "Run this script again after correcting the paths."
    return 1
fi

echo "All dependencies found! [OK]"
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "=== Configuration Summary ==="
echo "MSYS2:        $MSYS2_PATH"
echo "Android SDK:  $ANDROID_SDK_ROOT"
echo "Android NDK:  $ANDROID_NDK_HOME"
echo "Java JDK:     $JAVA_HOME"
echo "Hexagon SDK:  $HEXAGON_SDK_ROOT"
echo "OpenCL:       [AUTO] Downloaded during build"
echo ""

# Save to .env file (OpenCL is auto-downloaded, no config needed)
CONFIG_PATH="$(dirname "$0")/.env"
cat > "$CONFIG_PATH" << EOF
MSYS2_PATH=$MSYS2_PATH
ANDROID_SDK_ROOT=$ANDROID_SDK_ROOT
ANDROID_NDK_HOME=$ANDROID_NDK_HOME
JAVA_HOME=$JAVA_HOME
HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT
EOF

echo "Configuration saved to: $CONFIG_PATH"
echo ""
echo "Next: Run ./buildAPK.sh -All -install"
echo ""
echo "NOTE: The .env file is automatically loaded by all build scripts."
echo "You do NOT need to run this script before each build."
