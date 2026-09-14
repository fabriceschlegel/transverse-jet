# AWS GPU trial

This folder prepares a first nekRS GPU smoke test. It does not yet contain the
body-fitted transverse-jet mesh or claim physical equivalence with the local MAC solver.

The intended instance is one `g6.2xlarge` in `us-east-1`, using an AWS Ubuntu GPU
image that already supplies an NVIDIA driver and CUDA compiler. Its current console
price is $0.9776 per running hour. The cloud-init script schedules an operating-system
shutdown after 90 minutes, which caps the first session's instance runtime near $1.47.
Stopping preserves the EBS volume and therefore continues a small storage charge.

At launch:

1. Use one instance and a 30 GiB gp3 root volume.
2. Restrict SSH to `My IP`, or use EC2 Instance Connect. Do not expose SSH to `0.0.0.0/0`.
3. Paste `user-data-nekrs.sh` into Advanced details / User data.
4. Keep the default shutdown behavior `Stop` for this first trial.
5. After launch, inspect `/var/log/transverse-jet/bootstrap.log` and verify
   `/var/log/transverse-jet/bootstrap-complete` exists.

The script installs build tools and OpenMPI, builds the tagged nekRS 26.0 release
with CUDA, and runs its bundled channel case on one GPU. A successful smoke test is
the gate before building the transverse-jet geometry.

The EC2 account needs a quota of at least 8 vCPUs for “Running On-Demand G and VT
instances.” The minimum quota request was submitted in `us-east-1` before launch.
The request itself is free; compute billing begins only after an instance is launched.
