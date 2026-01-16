# Pure Fields of Small Discriminant

This project is a follow up to my [project on quadratic fields](https://github.com/algebraity/quadratic-field-analysis) where I did something similar. My goal is to compute the invariants all all pure, totally real fields with a minimal polynomial of degree p from 2 to some small prime, with discriminant less than some large number that is still computationally feasible.

I learned some interesting things from my previous project, but now that I know more about number fields, I would like to track more invariants for more fields and see what patterns I can find. This would also make a good entry level data engineering project once I have all the numbers, which will enable me to use my fledgling data analysis skills in an interesting way.

# Roadmap

My first course of action is to figure out what all I want to compute. Once I have a list, I will test it out on totally real quadratics in a way similar to what I did before. Then, I will generalize the code and allow and prime and a bound on the discriminant to be entered, so I can compute these invariants for much larger fields. I will then slowly compute invariants for larger and larger fields, gathering an extensive data set in the process.

Checklist:
1. Decide list of invariants to compute and the methods by which to compute them.
2. Appropriate my code from [my previous project](https://github.com/algebraity/quadratic-field-analysis) to compute these new invariants.
3. Generalize the code so I can compute with a bound on the discriminant, and use any degree.
4. Start generating the data sets.

# Usage

## Computing Fields

```bash
# Basic computation
sage -python compute_fields.py 4 10000000 -j 8

# Resume after crash (automatic)
sage -python compute_fields.py 4 10000000 -j 8

# Start fresh, ignore checkpoint
sage -python compute_fields.py 4 10000000 -j 8 --no-resume

# Skip expensive Galois computation
sage -python compute_fields.py 5 100000000 -j 16 --no-galois

# All options
sage -python compute_fields.py --help
```

## Querying Data

```bash
# Statistics
python query_fields.py data --stats

# Class number distribution
python query_fields.py data --class-dist -d 4

# Galois group distribution
python query_fields.py data --galois-dist -d 4

# Custom SQL
python query_fields.py data --sql "SELECT * FROM fields WHERE hk > 1"

# Export to CSV
python query_fields.py data --export fields.csv
```

## Output Format

Data is stored in Parquet format with zstd compression. Schema:

| Field | Type | Description |
|-------|------|-------------|
| poly_coeffs | list[int] | Defining polynomial coefficients |
| degree | int | Field degree |
| dk | int | Discriminant |
| hk | int | Class number |
| rk | float | Regulator |
| mb | float | Minkowski bound |
| signature | list[int] | (r1, r2) |
| fund_units | list[str] | Fundamental units |
| torsion_order | int | Torsion subgroup order |
| ramified_primes | str | JSON-encoded ramification data |
| int_basis | list[str] | Integral basis |
| auto_count | int | Number of automorphisms |
| galois_label | str | Galois group label (e.g. "4T3") |
| is_galois | bool | Whether field is Galois |

# Requirements

- SageMath
- pyarrow
- duckdb (for queries)
