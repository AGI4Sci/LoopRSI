from __future__ import annotations

import json

import official_h1_candidate_j_external_basis as cj
import official_h1_candidate_o_target_complete as candidate_o


if __name__ == "__main__":
    candidate_o.PROTOCOL_ID = "vcc-h1-candidate-p-genomewide-signature-transport-v1"
    print(json.dumps(candidate_o.run(cj.parse_args()), indent=2, sort_keys=True))
