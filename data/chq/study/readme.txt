This directory contains reader study results for the patient questions dataset.

- results.xlsx: reader scores exported from qualtrics
	- rows correspond to individual readers
	- columns {sample_idx}_{question_id} correspond to:
		- sample_idx: a particular sample
		- question_id {1, 2, 3} maps to {completeness, correctness, conciseness}
- blind_order.json: tracks the blinded order in which output, target were presented
	- if switch, multiply scores by -1
	- then positive scores denote output > target
