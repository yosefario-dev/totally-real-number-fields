#!/usr/bin/env python3
"""Query utilities for number field datasets in Parquet format."""

import duckdb
from pathlib import Path
from typing import Optional, Union
import pandas as pd


class FieldDataset:
    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.data_dir = Path(data_dir)
        self.con = duckdb.connect()
        self._register_views()
    
    def _register_views(self):
        pattern = str(self.data_dir / "*.parquet")
        self.con.execute(f"CREATE OR REPLACE VIEW fields AS SELECT * FROM '{pattern}'")
    
    def query(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()
    
    def count(self, degree: Optional[int] = None) -> int:
        where = f"WHERE degree = {degree}" if degree else ""
        result = self.con.execute(f"SELECT COUNT(*) FROM fields {where}").fetchone()
        return result[0]
    
    def class_number_distribution(self, degree: Optional[int] = None) -> pd.DataFrame:
        where = f"WHERE degree = {degree}" if degree else ""
        return self.query(f"""
            SELECT hk, COUNT(*) as count 
            FROM fields {where}
            GROUP BY hk 
            ORDER BY hk
        """)
    
    def galois_distribution(self, degree: Optional[int] = None) -> pd.DataFrame:
        where = f"WHERE degree = {degree}" if degree else ""
        return self.query(f"""
            SELECT galois_label, COUNT(*) as count
            FROM fields {where}
            GROUP BY galois_label
            ORDER BY count DESC
        """)
    
    def large_class_numbers(self, min_hk: int = 2, limit: int = 100) -> pd.DataFrame:
        return self.query(f"""
            SELECT poly_coeffs, dk, hk, galois_label, is_galois
            FROM fields
            WHERE hk >= {min_hk}
            ORDER BY hk DESC, dk ASC
            LIMIT {limit}
        """)
    
    def by_discriminant_range(self, dk_min: int, dk_max: int) -> pd.DataFrame:
        return self.query(f"""
            SELECT * FROM fields
            WHERE dk >= {dk_min} AND dk < {dk_max}
            ORDER BY dk
        """)
    
    def statistics(self, degree: Optional[int] = None) -> pd.DataFrame:
        where = f"WHERE degree = {degree}" if degree else ""
        return self.query(f"""
            SELECT 
                degree,
                COUNT(*) as total_fields,
                MIN(dk) as min_dk,
                MAX(dk) as max_dk,
                AVG(hk) as avg_class_number,
                MAX(hk) as max_class_number,
                AVG(rk) as avg_regulator,
                SUM(CASE WHEN is_galois THEN 1 ELSE 0 END) as galois_fields
            FROM fields {where}
            GROUP BY degree
            ORDER BY degree
        """)
    
    def export_csv(self, output_path: str, degree: Optional[int] = None):
        where = f"WHERE degree = {degree}" if degree else ""
        self.con.execute(f"""
            COPY (SELECT * FROM fields {where} ORDER BY dk) 
            TO '{output_path}' (HEADER, DELIMITER ',')
        """)
    
    def sample(self, n: int = 10, degree: Optional[int] = None) -> pd.DataFrame:
        where = f"WHERE degree = {degree}" if degree else ""
        return self.query(f"SELECT * FROM fields {where} USING SAMPLE {n}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Query number field dataset')
    parser.add_argument('data_dir', nargs='?', default='data', help='Data directory')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--class-dist', action='store_true', help='Class number distribution')
    parser.add_argument('--galois-dist', action='store_true', help='Galois group distribution')
    parser.add_argument('--degree', '-d', type=int, help='Filter by degree')
    parser.add_argument('--sql', type=str, help='Custom SQL query')
    parser.add_argument('--export', type=str, help='Export to CSV')
    
    args = parser.parse_args()
    ds = FieldDataset(args.data_dir)
    
    if args.sql:
        print(ds.query(args.sql).to_string())
    elif args.stats:
        print(ds.statistics(args.degree).to_string())
    elif args.class_dist:
        print(ds.class_number_distribution(args.degree).to_string())
    elif args.galois_dist:
        print(ds.galois_distribution(args.degree).to_string())
    elif args.export:
        ds.export_csv(args.export, args.degree)
        print(f"Exported to {args.export}")
    else:
        print(f"Total fields: {ds.count(args.degree)}")
        print("\nSample:")
        print(ds.sample(5, args.degree).to_string())


if __name__ == "__main__":
    main()
