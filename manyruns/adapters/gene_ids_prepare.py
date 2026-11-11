"""Prepare an HGNC archive as an unambiguous symbol-to-Ensembl mapping."""
import argparse
import re


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.url, sep='\t', dtype=str, keep_default_na=False)
    candidates, approved = {}, {}
    for row in source.to_dict('records'):
        gene_id = re.sub(r'\.\d+$', '', row['ensembl_gene_id'])
        if not re.fullmatch(r'ENSG\d{11}', gene_id):
            continue
        approved[row['symbol']] = gene_id
        symbols = [row['symbol'], *row.get('prev_symbol', '').split('|'), *row.get('alias_symbol', '').split('|')]
        for symbol in filter(None, symbols):
            candidates.setdefault(symbol, set()).add(gene_id)
    mapping = {symbol: next(iter(ids)) for symbol, ids in candidates.items() if len(ids) == 1}
    mapping.update(approved)
    pd.DataFrame(sorted(mapping.items()), columns=['symbol', 'gene_id']).to_csv(args.out, sep='\t', index=False)


if __name__ == '__main__':
    main()
