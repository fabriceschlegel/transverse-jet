#!/usr/bin/env bash
set -Eeuo pipefail

# EC2 cloud-init bootstrap for an Ubuntu NVIDIA GPU image.
# Stop after 90 minutes even if the interactive session is lost.
shutdown -h +90

install_root=/opt/nekrs
source_root=/opt/src/nekRS
log_root=/var/log/transverse-jet
mkdir -p "$log_root" /opt/src
exec > >(tee -a "$log_root/bootstrap.log") 2>&1

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  ca-certificates \
  cmake \
  gfortran \
  git \
  libopenmpi-dev \
  openmpi-bin

if ! command -v nvidia-smi >/dev/null; then
  echo "The selected image does not provide an NVIDIA driver." >&2
  exit 1
fi

if ! command -v nvcc >/dev/null; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
  else
    echo "The selected image does not provide the CUDA compiler required to build nekRS." >&2
    exit 1
  fi
fi

nvidia-smi
nvcc --version

if [[ ! -d "$source_root/.git" ]]; then
  git clone --branch v26.0 --depth 1 https://github.com/Nek5000/nekRS.git "$source_root"
fi

cd "$source_root"
printf '\n' | env CC=mpicc CXX=mpic++ FC=mpifort \
  ./build.sh -DCMAKE_INSTALL_PREFIX="$install_root"

cat > /etc/profile.d/nekrs.sh <<'EOF'
export NEKRS_HOME=/opt/nekrs
export PATH=/opt/nekrs/bin:$PATH
EOF

mkdir -p /home/ubuntu/benchmarks
cp -a "$install_root/examples/channel" /home/ubuntu/benchmarks/
chown -R ubuntu:ubuntu /home/ubuntu/benchmarks

sudo -u ubuntu env NEKRS_HOME="$install_root" PATH="$install_root/bin:$PATH" \
  bash -lc 'cd /home/ubuntu/benchmarks/channel && nrsmpi channel 1'

date --iso-8601=seconds > "$log_root/bootstrap-complete"
echo "nekRS installation and one-GPU channel smoke test completed."
