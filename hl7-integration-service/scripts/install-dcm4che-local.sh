#!/usr/bin/env bash
set -euo pipefail

# Installs DCM4CHE jars into local Maven repo if not available
# Usage: ./install-dcm4che-local.sh

GROUP=org.dcm4che
GROUP_PATH=${GROUP//./\/}
ARTIFACT_CORE=dcm4che-core
ARTIFACT_NET=dcm4che-net
VERSION=5.23.0

# Paths to possible existing jars in local m2 (cache)
LOCAL_M2="$HOME/.m2/repository/$GROUP_PATH"
CORE_JAR="$LOCAL_M2/$ARTIFACT_CORE/$VERSION/$ARTIFACT_CORE-$VERSION.jar"
NET_JAR="$LOCAL_M2/$ARTIFACT_NET/$VERSION/$ARTIFACT_NET-$VERSION.jar"

install_if_missing() {
  local artifact=$1
  local jarpath=$2
  local version=$3

  if mvn dependency:get -Dartifact=$GROUP:$artifact:$version >/dev/null 2>&1; then
    echo "$artifact:$version already available (or downloadable). Skipping local install."
    return
  fi

  if [ -f "$jarpath" ]; then
    echo "$artifact jar present at $jarpath. Checking local repository status..."
    # If artifact is already installed locally, skip install
    if [ -f "$jarpath" ]; then
      echo "Skipping install: $artifact $version already in local repository (jar present)."
      return
    fi
    echo "Installing $artifact from $jarpath into local repo as $version"
    mvn install:install-file \
      -Dfile="$jarpath" \
      -DgroupId="$GROUP" \
      -DartifactId="$artifact" \
      -Dversion="$version" \
      -Dpackaging=jar \
      -DgeneratePom=true || true
  else
    echo "ERROR: $jarpath not found. Please download $artifact-$version.jar and place it at $jarpath or add to libs/ and re-run this script."
    exit 1
  fi
}

install_if_missing "$ARTIFACT_CORE" "$CORE_JAR" "$VERSION"
install_if_missing "$ARTIFACT_NET" "$NET_JAR" "$VERSION"

echo "Done. You can now run: mvn -o -DskipTests clean package"