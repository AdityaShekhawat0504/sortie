## Detection on injected faults

| Injected | Expected | Trials | Detected | Median latency | Other false alarms / trial |
|---|---|---:|---:|---:|---:|
| `ekf_rejection` | estimator | 125 | 100% | 0.02 s | 0.00 |
| `gps_jump` | gps | 43 | 100% | 0.03 s | 0.00 |
| `gps_degradation` | gps | 49 | 100% | 0.00 s | 0.00 |
| `vibration_burst` | vibration | 50 | 100% | 0.00 s | 0.00 |
| `battery_sag` | battery | 50 | 72% | 0.48 s | 0.00 |
| `motor_failure` | motors | 50 | 100% | 0.06 s | 0.00 |
| `motor_saturation` | motors | 50 | 100% | 0.05 s | 0.00 |
| `control_load_step` | nothing (healthy control) | 50 | 0% flagged | n/a | 0.00 |
| `attitude_divergence` | attitude | 25 | 92% | 0.30 s | 0.00 |
| `control_tracked_manoeuvre` | nothing (healthy control) | 25 | 0% flagged | n/a | 0.00 |

## Smallest fault detected in >= 50% of trials

| Fault | Parameter | Detected at >= |
|---|---|---:|
| `gps_jump` | jump [m] | 12.73 |
| `gps_degradation` | eph [m] | 3.54 |
| `vibration_burst` | noise sigma [m/s^2] | 4.68 |
| `battery_sag` | drop [V/cell] | 0.34 |
| `attitude_divergence` | error [deg] | 11.00 |
| `ekf_rejection` | test ratio | 1.04 |

## Unsupervised layer: injected fault among top-3 anomalous seconds

              fault  trials  in_top3  random_baseline
           gps_jump      30    0.800            0.500
    gps_degradation      30    0.633            0.500
    vibration_burst      15    0.800            0.400
        battery_sag      15    1.000            0.400
      motor_failure      15    1.000            0.400
   motor_saturation      15    1.000            0.400
attitude_divergence      15    1.000            0.400
      ekf_rejection      60    0.617            0.285
