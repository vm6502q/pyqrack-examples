OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
h q[0];
cp(pi / 2) q[1], q[0];
h q[1];
cp(pi / 4) q[2], q[0];
cp(pi / 2) q[2], q[1];
h q[2];
cp(pi / 8) q[3], q[0];
cp(pi / 4) q[3], q[1];
cp(pi / 2) q[3], q[2];
h q[3];
