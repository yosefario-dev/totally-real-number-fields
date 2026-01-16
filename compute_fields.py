#!/usr/bin/env sage -python
"""
Compute invariants of totally real number fields and store in Parquet format.
Supports parallel computation, crash recovery, and single-file output.
"""

from sage.all import *
from multiprocessing import Pool, cpu_count
import pyarrow as pa
import pyarrow.parquet as pq
import json
import os
import sys
import logging
import time
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

SCHEMA = pa.schema([
    ('poly_coeffs', pa.list_(pa.int64())),
    ('degree', pa.uint8()),
    ('dk', pa.int64()),
    ('hk', pa.uint32()),
    ('rk', pa.float64()),
    ('mb', pa.float64()),
    ('signature', pa.list_(pa.uint8())),
    ('fund_units', pa.list_(pa.string())),
    ('torsion_order', pa.uint8()),
    ('ramified_primes', pa.string()),
    ('int_basis', pa.list_(pa.string())),
    ('auto_count', pa.uint8()),
    ('galois_label', pa.string()),
    ('is_galois', pa.bool_()),
])

CHECKPOINT_INTERVAL = 60


@dataclass
class ComputeConfig:
    degree: int
    dk_bound: int
    output_dir: Path
    batch_size: int = 1000
    compute_galois: bool = True
    n_workers: Optional[int] = None


class CheckpointManager:
    def __init__(self, output_dir: Path, degree: int):
        self.checkpoint_file = output_dir / f".checkpoint_deg{degree}.json"
        self.last_save = time.time()
    
    def load(self) -> Dict[str, Any]:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {"last_dk": 0, "total_computed": 0, "batch_num": 0}
    
    def save(self, last_dk: int, total_computed: int, batch_num: int, force: bool = False):
        now = time.time()
        if not force and (now - self.last_save) < CHECKPOINT_INTERVAL:
            return
        
        data = {
            "last_dk": last_dk,
            "total_computed": total_computed,
            "batch_num": batch_num,
            "timestamp": now
        }
        
        tmp = self.checkpoint_file.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(data, f)
        tmp.rename(self.checkpoint_file)
        self.last_save = now
    
    def clear(self):
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


class ParquetAppender:
    """Incrementally append batches to a single Parquet file."""
    
    def __init__(self, output_path: Path, schema: pa.Schema):
        self.output_path = output_path
        self.schema = schema
        self.writer: Optional[pq.ParquetWriter] = None
        self.rows_written = 0
        self.logger = logging.getLogger(__name__)
        
        self.temp_path = output_path.with_suffix('.parquet.tmp')
    
    def _ensure_writer(self):
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.temp_path,
                self.schema,
                compression='zstd'
            )
    
    def write_batch(self, records: List[Dict]):
        if not records:
            return
        
        self._ensure_writer()
        
        arrays = {field.name: [] for field in self.schema}
        for rec in records:
            for key in arrays:
                arrays[key].append(rec[key])
        
        table = pa.table(arrays, schema=self.schema)
        self.writer.write_table(table)
        self.rows_written += len(records)
        
        self.logger.debug(f"Wrote batch of {len(records)} records, total {self.rows_written}")
    
    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            
            if self.temp_path.exists():
                self.temp_path.rename(self.output_path)
                self.logger.info(f"Finalized output: {self.output_path}")


def setup_logging(output_dir: Path, verbose: bool = False):
    log_file = output_dir / "compute.log"
    
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)


def compute_single_field(poly_data, compute_galois: bool = True) -> Optional[Dict]:
    try:
        dk, poly_coeffs = poly_data
        R = PolynomialRing(QQ, 'x')
        poly = R(poly_coeffs)
        K = NumberField(poly, 'u')
        degree = K.degree()
        
        hK = K.class_number()
        rK = float(K.regulator())
        mb = float(K.minkowski_bound())
        sig = list(K.signature())
        
        U = K.unit_group()
        fu = [str(u) for u in U.fundamental_units()]
        tor_order = U.torsion_subgroup().order()
        
        ram_data = {}
        for p in K.discriminant().prime_factors():
            ideal = K.ideal(p)
            ram_data[int(p)] = [
                [str(f), int(f.ramification_index())] 
                for f in ideal.prime_factors()
            ]
        
        int_basis = [str(b) for b in K.integral_basis()]
        
        autos = K.automorphisms()
        auto_count = len(autos)
        
        galois_label = ""
        is_galois = (auto_count == degree)
        
        if compute_galois:
            try:
                G = K.galois_group()
                label_str = str(G)
                if '(' in label_str:
                    galois_label = label_str.split('(')[1].split(')')[0]
                else:
                    parts = label_str.split()
                    galois_label = parts[2] if len(parts) > 2 else "?"
            except Exception:
                galois_label = "?"
        
        return {
            'poly_coeffs': list(poly_coeffs),
            'degree': degree,
            'dk': int(dk),
            'hk': int(hK),
            'rk': rK,
            'mb': mb,
            'signature': sig,
            'fund_units': fu,
            'torsion_order': int(tor_order),
            'ramified_primes': json.dumps(ram_data),
            'int_basis': int_basis,
            'auto_count': auto_count,
            'galois_label': galois_label,
            'is_galois': is_galois,
        }
    except Exception:
        return None


def compute_batch(fields_batch: List, config: ComputeConfig) -> List[Dict]:
    n_workers = config.n_workers or max(1, cpu_count() - 1)
    worker_fn = partial(compute_single_field, compute_galois=config.compute_galois)
    
    with Pool(n_workers) as pool:
        results = pool.map(worker_fn, fields_batch)
    
    return [r for r in results if r is not None]


def merge_existing_data(output_path: Path, writer: ParquetAppender, last_dk: int, logger):
    """Load existing data up to last_dk into the new writer."""
    temp_existing = output_path.with_suffix('.parquet.old')
    
    if not output_path.exists():
        return
    
    logger.info(f"Merging existing data up to dk={last_dk}")
    output_path.rename(temp_existing)
    
    try:
        table = pq.read_table(temp_existing)
        df = table.to_pandas()
        preserved = df[df['dk'] <= last_dk]
        
        if len(preserved) > 0:
            records = preserved.to_dict('records')
            for rec in records:
                rec['poly_coeffs'] = list(rec['poly_coeffs'])
                rec['signature'] = list(rec['signature'])
                rec['fund_units'] = list(rec['fund_units'])
                rec['int_basis'] = list(rec['int_basis'])
            writer.write_batch(records)
            logger.info(f"Preserved {len(preserved)} existing records")
        
        temp_existing.unlink()
    except Exception as e:
        logger.error(f"Failed to merge existing data: {e}")
        if temp_existing.exists():
            temp_existing.rename(output_path)
        raise


def compute_fields(config: ComputeConfig, resume: bool = True, verbose: bool = False):
    config.output_dir = Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(config.output_dir, verbose)
    checkpoint = CheckpointManager(config.output_dir, config.degree)
    
    output_file = config.output_dir / f"fields_deg{config.degree}_dk{config.dk_bound}.parquet"
    writer = ParquetAppender(output_file, SCHEMA)
    
    state = {"last_dk": 0, "total_computed": 0, "batch_num": 0}
    
    if resume:
        state = checkpoint.load()
        if state["last_dk"] > 0:
            logger.info(f"Resuming from dk={state['last_dk']}, {state['total_computed']} fields")
            merge_existing_data(output_file, writer, state["last_dk"], logger)
    
    logger.info(f"Computing degree {config.degree} fields with dk < {config.dk_bound}")
    logger.info(f"Workers: {config.n_workers or cpu_count()-1}, Batch size: {config.batch_size}")
    logger.info(f"Galois computation: {'enabled' if config.compute_galois else 'disabled'}")
    logger.info(f"Output: {output_file}")
    
    start_time = time.time()
    fields_gen = enumerate_totallyreal_fields_prim(config.degree, config.dk_bound)
    
    batch = []
    skipped = 0
    
    try:
        for field_data in fields_gen:
            dk = field_data[0]
            
            if dk <= state["last_dk"]:
                skipped += 1
                continue
            
            if skipped > 0:
                logger.info(f"Skipped {skipped} already-computed fields")
                skipped = 0
            
            batch.append(field_data)
            
            if len(batch) >= config.batch_size:
                results = compute_batch(batch, config)
                
                if results:
                    writer.write_batch(results)
                    state["total_computed"] += len(results)
                    state["last_dk"] = max(r['dk'] for r in results)
                    
                    elapsed = time.time() - start_time
                    rate = state["total_computed"] / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Batch {state['batch_num']}: {len(results)} fields, "
                        f"dk={state['last_dk']}, "
                        f"total={state['total_computed']} ({rate:.1f}/s)"
                    )
                
                checkpoint.save(state["last_dk"], state["total_computed"], state["batch_num"])
                batch = []
                state["batch_num"] += 1
        
        if batch:
            results = compute_batch(batch, config)
            if results:
                writer.write_batch(results)
                state["total_computed"] += len(results)
                state["last_dk"] = max(r['dk'] for r in results)
        
        writer.close()
        checkpoint.save(state["last_dk"], state["total_computed"], state["batch_num"], force=True)
        checkpoint.clear()
        
        elapsed = time.time() - start_time
        logger.info(f"Completed: {state['total_computed']} fields in {elapsed:.1f}s")
        logger.info(f"Output: {output_file}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted, saving progress...")
        writer.close()
        checkpoint.save(state["last_dk"], state["total_computed"], state["batch_num"], force=True)
        logger.info(f"Checkpoint saved at dk={state['last_dk']}")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        writer.close()
        checkpoint.save(state["last_dk"], state["total_computed"], state["batch_num"], force=True)
        raise
    
    return state["total_computed"]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute number field invariants',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('degree', type=int, help='Degree of fields')
    parser.add_argument('bound', type=int, help='Discriminant bound')
    parser.add_argument('-j', '--workers', type=int, default=None, help='Parallel workers')
    parser.add_argument('-b', '--batch-size', type=int, default=1000, help='Batch size')
    parser.add_argument('--no-galois', action='store_true', help='Skip Galois groups')
    parser.add_argument('-o', '--output', default='data', help='Output directory')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    config = ComputeConfig(
        degree=args.degree,
        dk_bound=args.bound,
        output_dir=Path(args.output),
        batch_size=args.batch_size,
        compute_galois=not args.no_galois,
        n_workers=args.workers,
    )
    
    compute_fields(config, resume=not args.no_resume, verbose=args.verbose)


if __name__ == "__main__":
    main()
