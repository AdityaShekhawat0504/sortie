# Sample logs

Real PX4 ULog files from the test suite of [PX4/pyulog](https://github.com/PX4/pyulog)
(BSD-3-Clause, see `PYULOG_LICENSE.md`), renamed by hardware and length.

| File | Hardware | What it is |
|---|---|---|
| `sitl_quad_31s.ulg` | PX4 SITL | Quadrotor takeoff and flight, armed 14.9 to 30.1 s |
| `cubeorange_vtol_6s.ulg` | Cube Orange | VTOL armed on the ground (land detector: on ground throughout) |
| `auav_x21_69s.ulg` | AUAV X2.1 | Bench log, never armed |
| `fmuv4_115s.ulg` | Pixracer (FMUv4) | Bench log, never armed, uncalibrated power module |
| `sitl_bench_11s.ulg` | PX4 SITL | Startup log, never armed, GPS still converging |

They span four PX4 schema generations, which is why `sortie.ingest` maps every
topic through an adapter.
