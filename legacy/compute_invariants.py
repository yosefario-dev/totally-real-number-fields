from sage.all import *

def compute_field(K):
    poly = K.defining_polynomial()
    dK = K.discriminant()
    C = K.class_group()
    hK = K.class_number()                           
    rK = K.regulator()                              
    mb = K.minkowski_bound()

    U = K.unit_group()
    fu = U.fundamental_units()
    tor = U.torsion_subgroup()
    
    primes = dK.prime_factors()
    prime_ram_data = {}
    for p in primes:
        i = K.ideal(p)
        p_data = []
        for factor in i.prime_factors():
            p_data.append([factor, factor.ramification_index()])
        prime_ram_data[p] = p_data

    int_base = K.integral_basis()

    autos = K.automorphisms()
    galois_grp = K.galois_group()
    galois_clsr = K.galois_closure('v')

    field_data = {}
    field_data["field"] = str(K)
    field_data["min_poly"] = poly
    field_data["dk"] = dK
    field_data["hk"] = hK
    field_data["rk"] = rK
    field_data["mb"] = mb
    field_data["class_group"] = C
    field_data["unit_group"] = U
    field_data["fund_units"] = fu
    field_data["unit_tor_sbgrp"] = tor
    field_data["ramified_primes"] = prime_ram_data
    field_data["int_basis"] = int_base
    field_data["auto_group"] = autos
    field_data["galois_group"] = galois_grp
    field_data["galois_clsr"] = galois_clsr

    return field_data


def compute_invariants(deg, d_bound):
    fields = enumerate_totallyreal_fields_prim(deg, d_bound)
    field_data = []
    for field in fields:
        R = PolynomialRing(QQ, 'x')
        poly = R(field[1])
        K = NumberField(poly, 'u')
        field_data.append(compute_field(K))
    return field_data
