# Multi-seed summary

Values are mean +/- sample std across seed artifacts.

## Inputs

- sequence files: 10
- controller anchor files: 10

## Pattern check

- Independent single-task fits average 1.000 final accuracy.
- Composition remains the failure point: coeff_add retention is 0.225, naive_stack is 0.662; unanchored composed controller is 0.300.
- Routing stays strong: unanchored routed 1.000, anchored routed 0.950.
- The drift anchor lowers controller drift from 5.090 to 0.606 (8.4x lower).
- Paraphrase consistency remains 0.000 on average across composed methods.

## Aggregate metrics

| condition | n | final acc avg | A | B | C | retention | rev gap | collateral | order sens | drift KL | paraphrase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 10 | 0.167 +/- 0.000 | 0.250 +/- 0.000 | 0.250 +/- 0.000 | 0.000 +/- 0.000 | - | - | - | - | 0.000 +/- 0.000 | - |
| independent | 10 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | - | - | - | - | 5.172 +/- 1.220 | - |
| naive_stack | 10 | 0.775 +/- 0.069 | 0.600 +/- 0.175 | 0.725 +/- 0.079 | 1.000 +/- 0.000 | 0.662 +/- 0.103 | 0.225 +/- 0.079 | 0.475 +/- 0.115 | 0.342 +/- 0.127 | 15.338 +/- 4.190 | 0.000 +/- 0.000 |
| coeff_add | 10 | 0.200 +/- 0.058 | 0.275 +/- 0.142 | 0.175 +/- 0.121 | 0.150 +/- 0.129 | 0.225 +/- 0.079 | 0.075 +/- 0.121 | 0.125 +/- 0.102 | 0.000 +/- 0.000 | 6.387 +/- 1.308 | 0.000 +/- 0.000 |
| controller | 10 | 0.533 +/- 0.119 | 0.250 +/- 0.167 | 0.350 +/- 0.242 | 1.000 +/- 0.000 | 0.300 +/- 0.179 | 0.050 +/- 0.105 | 0.600 +/- 0.079 | 0.575 +/- 0.133 | 5.090 +/- 0.996 | 0.000 +/- 0.000 |
| controller_anchor | 10 | 0.475 +/- 0.056 | 0.200 +/- 0.158 | 0.225 +/- 0.079 | 1.000 +/- 0.000 | 0.212 +/- 0.084 | 0.050 +/- 0.105 | 0.537 +/- 0.084 | 0.633 +/- 0.081 | 0.606 +/- 0.122 | 0.000 +/- 0.000 |
| controller_routed | 10 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | - | - | - | - | - | - |
| controller_anchor_routed | 10 | 0.950 +/- 0.090 | 0.850 +/- 0.269 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | - | - | - | - | - | - |

Notes:

- `independent` drift KL is the maximum single-task drift; it has no single composed final state.
- `controller_routed` rows evaluate controller-predicted standalone coefficients from held-out context phrasings, so retention, reversibility, order sensitivity, drift, and paraphrase are not defined for those rows.
