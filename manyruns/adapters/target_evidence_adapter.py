"""Join ranked genes to a prepared disease resource without network access."""
import hashlib
import json
from pathlib import Path
import sys


def main(request_path):
    import numpy as np
    import pandas as pd

    req = json.loads(Path(request_path).read_text())
    params = req['params']
    disease = params.get('disease', '')
    if not disease or disease in ('.', '..') or any(c in disease for c in '/\\\x00'):
        Path(req['result']).write_text(json.dumps({'ok': False, 'reason': 'disease must be a non-empty file-safe key'}))
        return 0
    genes = pd.read_csv(req['tables']['genes'], sep='\t', dtype={'gene': str, 'gene_id': str}, keep_default_na=False)
    genes['score'] = pd.to_numeric(genes.score, errors='coerce')
    genes = genes.loc[np.isfinite(genes.score)].copy()
    genes = genes.loc[genes.score.abs().sort_values(ascending=False, kind='stable').index].head(params.get('top_n', 100))
    evidence = pd.read_csv(req['resources']['opentargets'], sep='\t', dtype={'gene_id': str}, keep_default_na=False)
    evidence = evidence.loc[(evidence.disease_id == disease) & (evidence.gene_id != '')]
    output = genes.merge(evidence, on='gene_id', how='inner', sort=False, validate='many_to_one')
    output = output[['gene_id', 'symbol', 'disease_id', 'assoc_score', 'gene', 'score']]
    path = Path(req['workdir']) / 'gene_evidence.tsv'
    output.to_csv(path, sep='\t', index=False)
    result = {'schema_version': 1, 'ok': True, 'n_cells': req['n_cells'], 'facts': {},
              'scalars': {'genes_queried': len(genes), 'genes_with_evidence': int(output.gene_id.nunique())},
              'tables': {'gene_evidence': {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
    Path(req['result']).write_text(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
