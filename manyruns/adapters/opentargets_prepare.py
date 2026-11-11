"""Filter the pinned association partitions to one disease during an explicit resource build."""
import argparse
from pathlib import Path
import shutil
import tempfile
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import urlopen

COLUMNS = ['gene_id', 'symbol', 'disease_id', 'assoc_score']


def prepare_frame(frame, key):
    selected = frame.loc[frame.diseaseId == key].rename(columns={
        'targetId': 'gene_id', 'diseaseId': 'disease_id', 'score': 'assoc_score',
    }).copy()
    # The overall association release has identifiers, but no gene symbols.
    if 'symbol' not in selected:
        selected['symbol'] = ''
    return selected[COLUMNS]


def sources(url):
    from html.parser import HTMLParser

    class Partitions(HTMLParser):
        def __init__(self):
            super().__init__()
            self.names = set()

        def handle_starttag(self, tag, attrs):
            href = dict(attrs).get('href', '')
            if tag == 'a' and href.endswith('.parquet') and Path(href).name == href:
                self.names.add(href)

    parsed = urlparse(url)
    if parsed.scheme == 'file':
        path = Path(unquote(parsed.path))
        if path.is_dir():
            paths = sorted(path.glob('*.parquet'))
            if not paths:
                raise ValueError('the dataset directory has no parquet partitions')
            return [p.as_uri() for p in paths]
        return [url]
    if parsed.path.endswith('.parquet'):
        return [url]
    with urlopen(url, timeout=60) as response:
        listing = Partitions()
        listing.feed(response.read().decode('utf-8'))
    if not listing.names:
        raise ValueError('the dataset URL has no parquet partitions')
    return [urljoin(url.rstrip('/') + '/', name) for name in sorted(listing.names)]


def main():
    import pandas as pd
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--key', required=True)
    args = parser.parse_args()
    if not args.key or args.key in ('.', '..') or any(c in args.key for c in '/\\\x00'):
        parser.error('key must be a non-empty file-safe disease identifier')
    with open(args.out, 'w') as output, tempfile.TemporaryDirectory() as temporary:
        pd.DataFrame(columns=COLUMNS).to_csv(output, sep='\t', index=False)
        for url in sources(args.url):
            # Only one remote partition occupies temporary storage at a time.
            parsed = urlparse(url)
            if parsed.scheme == 'file':
                path = Path(unquote(parsed.path))
            else:
                path = Path(temporary) / 'partition.parquet'
                with urlopen(url, timeout=60) as response, path.open('wb') as target:
                    shutil.copyfileobj(response, target)
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(columns=['targetId', 'diseaseId', 'score']):
                    prepare_frame(batch.to_pandas(), args.key).to_csv(output, sep='\t', index=False, header=False)


if __name__ == '__main__':
    main()
