#!/bin/sh
# Shared TLC discovery for exhaustive models and observed traces.

: "${repo_root:?source after locating the repository root}"

find_tla_jar() {
    if [ -n "${TLA2TOOLS_JAR:-}" ] && [ -f "$TLA2TOOLS_JAR" ]; then
        printf '%s\n' "$TLA2TOOLS_JAR"
        return
    fi

    if [ -f "$repo_root/.tools/tla2tools.jar" ]; then
        printf '%s\n' "$repo_root/.tools/tla2tools.jar"
        return
    fi

    toolbox_jar="/Applications/TLA+ Toolbox.app/Contents/Eclipse/tla2tools.jar"
    if [ -f "$toolbox_jar" ]; then
        printf '%s\n' "$toolbox_jar"
        return
    fi

    printf '%s\n' \
        "TLC not found. Run 'make tla-tools' or set TLA2TOOLS_JAR." >&2
    exit 1
}

find_java() {
    if [ -n "${TLA_JAVA:-}" ] && [ -x "$TLA_JAVA" ]; then
        printf '%s\n' "$TLA_JAVA"
        return
    fi

    if command -v java >/dev/null 2>&1 && java -version >/dev/null 2>&1; then
        command -v java
        return
    fi

    for candidate in \
        /Applications/TLA+\ Toolbox.app/Contents/Eclipse/plugins/org.lamport.openjdk.*/Contents/Home/bin/java
    do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    printf '%s\n' \
        "Java 11 or newer is required to run TLC. Set TLA_JAVA if needed." >&2
    exit 1
}
