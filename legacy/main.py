from sage.all import *
import json
from compute_invariants import *

def main(deg=-1, d_bound=10**6):
    if deg == -1:
        deg = int(input("Enter the degree of the fields to compute: "))
        while not (isinstance(deg, int)) and not (n > 0):
            n = int(input("Input not accepted. Enter 0 for real and 1 for imaginary: "))
        d_bound = int(input("Enter a bound for the discriminant: "))

    invariants = compute_invariants(deg, d_bound)

    file_name = "field_data_deg_" + str(deg) + "_d-bound_" + str(d_bound) + ".json"
    to_json(invariants, file_name)

def serialize_field(field_data):
    out = {}

    out["field"] = str(field_data["field"])
    out["min_poly"] = str(field_data["min_poly"])
    out["dk"] = int(field_data["dk"])
    out["hk"] = int(field_data["hk"])

    out["rk"] = float(field_data["rk"])
    out["mb"] = float(field_data["mb"])

    out["class_group"] = str(field_data["class_group"])
    out["unit_group"] = str(field_data["unit_group"])

    out["fund_units"] = [str(u) for u in field_data["fund_units"]]
    out["unit_tor_sbgrp"] = str(field_data["unit_tor_sbgrp"])

    out["ramified_primes"] = {
        int(p): [[str(I), int(e)] for (I, e) in data]
        for p, data in field_data["ramified_primes"].items()
    }

    out["int_basis"] = [str(b) for b in field_data["int_basis"]]

    out["auto_group"] = [str(phi) for phi in field_data["auto_group"]]

    out["galois_group"] = str(field_data["galois_group"])
    out["galois_clsr"] = str(field_data["galois_clsr"])

    return out


def to_json(invariants, filename):
    print("Saving data to", filename)
    to_store = [serialize_field(field) for field in invariants]

    with open(filename, 'w') as json_file:
        json.dump(to_store, json_file, indent=2)

if __name__ == "__main__":
    main()
