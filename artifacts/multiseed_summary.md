# Multi-seed summary

Values are mean +/- sample std across seed artifacts.

## Inputs

- sequence files: 10
- controller anchor files: 10

## Pattern check

- Independent single-task fits average 1.000 final accuracy.
- Composition remains the failure point: coeff_add retention is 0.225, naive_stack is 0.613; unanchored composed controller is 0.300.
- Routing stays strong: unanchored routed 0.983, anchored routed 0.983.
- The drift anchor lowers controller drift from 4.949 to 0.615 (8.0x lower).
- Paraphrase consistency remains 0.431 on average across composed methods.

## Aggregate metrics

| condition | n | final acc avg | A | B | C | retention | cram (newest alone) | rev gap | collateral | order sens | drift KL | paraphrase | heldout acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 10 | 0.167 +/- 0.000 | 0.250 +/- 0.000 | 0.250 +/- 0.000 | 0.000 +/- 0.000 | - | - | - | - | - | 0.000 +/- 0.000 | - | - |
| independent | 10 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | - | - | - | - | - | 5.180 +/- 1.220 | - | - |
| naive_stack | 10 | 0.742 +/- 0.047 | 0.525 +/- 0.079 | 0.700 +/- 0.105 | 1.000 +/- 0.000 | 0.613 +/- 0.071 | - | 0.225 +/- 0.079 | 0.463 +/- 0.132 | 0.392 +/- 0.097 | 15.298 +/- 4.318 | 0.883 +/- 0.081 | - |
| coeff_add | 10 | 0.200 +/- 0.058 | 0.275 +/- 0.142 | 0.175 +/- 0.121 | 0.150 +/- 0.129 | 0.225 +/- 0.079 | - | 0.075 +/- 0.121 | 0.125 +/- 0.102 | 0.000 +/- 0.000 | 6.446 +/- 1.286 | 0.633 +/- 0.193 | - |
| controller | 10 | 0.533 +/- 0.081 | 0.225 +/- 0.142 | 0.375 +/- 0.243 | 1.000 +/- 0.000 | 0.300 +/- 0.121 | 0.175 +/- 0.087 | 0.100 +/- 0.129 | 0.550 +/- 0.147 | 0.650 +/- 0.077 | 4.949 +/- 0.922 | 0.117 +/- 0.125 | - |
| controller_anchor | 10 | 0.450 +/- 0.043 | 0.175 +/- 0.169 | 0.175 +/- 0.121 | 1.000 +/- 0.000 | 0.175 +/- 0.065 | 0.200 +/- 0.087 | 0.025 +/- 0.079 | 0.550 +/- 0.087 | 0.633 +/- 0.090 | 0.615 +/- 0.130 | 0.092 +/- 0.092 | - |
| controller_routed | 10 | 0.983 +/- 0.035 | 0.975 +/- 0.079 | 1.000 +/- 0.000 | 0.975 +/- 0.079 | - | - | - | - | - | - | - | - |
| controller_anchor_routed | 10 | 0.983 +/- 0.035 | 0.950 +/- 0.105 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | - | - | - | - | - | - | - | - |

Notes:

- `independent` drift KL is the maximum single-task drift; it has no single composed final state.
- `controller_routed` rows evaluate controller-predicted standalone coefficients from held-out context phrasings, so retention, reversibility, order sensitivity, drift, and paraphrase are not defined for those rows.
